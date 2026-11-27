"""The day's values at Europe's monitoring stations, from the EEA's up-to-date hours (E2a): the
measurements that correct the map's days after the archive (Panache's D90, D92).

The EEA's Air Quality Download Service serves, beside the validated hours (E1a, `stations`), the
hours the national networks stream as they are measured, unverified (dataset 1, "E2a" or "UTD").
Each sampling point has one Parquet file of the current year's hours, rewritten through the day;
the service lists the same files whatever the days asked, and a year's files go once its validated
hours are reported (by 30 September of the next year). Hours are later marked "preliminary
verified" (2) or "verified" (1) by some networks; the rest stay "not verified" (3).

Each file is read as `stations` reads the validated ones, the same UTC days, the same statistics
and the same rule on hours, after a screening of the hours (`Screening`) that the validated data do
not need. A day's values of every country make one asset per pollutant, of the validated assets'
schema and metadata, each point's country in its own entry.
"""

import json
import logging
import threading
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

from panache_archive import web
from panache_archive.stations import (
    DATASET_URL,
    DOWNLOAD_API,
    DOWNLOADS,
    MIN_HOURS,
    USER_AGENT,
    Point,
    PointDays,
    Pollutant,
    Read,
    daily_means,
    list_files,
    read_points,
    read_station_metadata,
    to_table,
)

UP_TO_DATE: Final = 1
"""The service's number for the up-to-date, unverified series (E2a)."""
ATTRIBUTION: Final = (
    "European Environment Agency, Air Quality e-Reporting (up-to-date data, E2a): unverified "
    "measurements of the national networks reported under Directive 2008/50/EC."
)
POLLUTANTS: Final = (Pollutant.PM2_5, Pollutant.PM10, Pollutant.OZONE)
"""The pollutants Panache's analysis corrects (D90)."""
CEILINGS: Final[Mapping[Pollutant, float]] = {
    Pollutant.PM2_5: 2000.0,
    Pollutant.PM10: 5000.0,
    Pollutant.OZONE: 800.0,
}
"""Hourly values no air in Europe reaches, µg/m³: about twice the highest hours of record (PM10 in
the Canary Islands' dust of February 2020, ozone in the 1970s' photochemical smog), so that only a
broken instrument or a wrong unit goes over."""
STUCK_HOURS: Final = 24
"""Consecutive hours of one same value that make it stuck: a whole day without a change."""
logger = logging.getLogger(__name__)


class Rule(StrEnum):
    """The screening's rules, in the order they are applied: an hour counts for the first one
    that removes it."""

    NEGATIVE = "negative"
    CEILING = "ceiling"
    STUCK = "stuck"


class Screening:
    """The screen of a pollutant's hours (`stations.Screen`), which counts what each rule removes
    on each UTC day. Safe to call from several threads."""

    def __init__(self, pollutant: Pollutant) -> None:
        self.ceiling = CEILINGS[pollutant]
        self.removed: Counter[tuple[Rule, str]] = Counter()
        self.kept: Counter[str] = Counter()
        self._lock = threading.Lock()

    def __call__(
        self, times: NDArray[np.datetime64], values: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        rules = {
            Rule.NEGATIVE: values < 0,
            Rule.CEILING: values > self.ceiling,
            Rule.STUCK: stuck(times, values),
        }
        keep = np.ones(values.size, np.bool_)
        days = times.astype("datetime64[D]").astype(str)
        with self._lock:
            for rule, hit in rules.items():
                removed = hit & keep
                self.removed.update((rule, day) for day in days[removed])
                keep &= ~hit
            self.kept.update(days[keep])
        return keep

    def of_day(self, day: str) -> dict[str, int]:
        """What each rule removed on `day`, and the values kept."""
        counts = {str(rule): self.removed[rule, day] for rule in Rule}
        return counts | {"kept": self.kept[day]}

    def total(self) -> dict[str, int]:
        """What each rule removed over every day, and the values kept."""
        counts = {str(rule): 0 for rule in Rule}
        for (rule, _), count in self.removed.items():
            counts[str(rule)] += count
        return counts | {"kept": sum(self.kept.values())}


def stuck(times: NDArray[np.datetime64], values: NDArray[np.float64]) -> NDArray[np.bool_]:
    """The values inside a run of `STUCK_HOURS` or more consecutive hours of one same value."""
    if not values.size:
        return np.zeros(0, np.bool_)
    step = np.diff(times) != np.timedelta64(1, "h")
    change = np.diff(values) != 0
    run = np.concatenate([[0], np.cumsum(step | change)])
    return np.asarray(np.bincount(run)[run] >= STUCK_HOURS, np.bool_)


@dataclass(frozen=True, slots=True)
class CountryDays:
    """A country's points and their daily values of a pollutant, screened."""

    country: str
    points: Mapping[str, Point]
    found: Sequence[PointDays]
    files: int


def list_countries(*, read: Read = web.read) -> list[str]:
    """The countries the download service has, by their two-letter codes."""
    answer = json.loads(read(f"{DOWNLOAD_API}/Country"))
    return sorted(str(entry["countryCode"]) for entry in answer)


def read_country(
    country: str,
    pollutant: Pollutant,
    metadata: bytes,
    screening: Screening,
    *,
    read: Read = web.read,
    downloads: int = DOWNLOADS,
) -> CountryDays:
    """Every daily value of `pollutant` the up-to-date files of `country` hold, screened."""
    # The listing does not depend on the days asked: one year lists every file.
    urls = list_files(country, pollutant, [date.today().year], read=read, dataset=UP_TO_DATE)
    with ThreadPoolExecutor(max_workers=downloads) as pool:
        found = list(
            pool.map(lambda url: daily_means(read(_get(url)), pollutant.statistic, screening), urls)
        )
    return CountryDays(country, read_points(metadata, country, pollutant), found, len(urls))


def asset_name(pollutant: Pollutant, day: date) -> str:
    return f"stations-live-{pollutant}-{day.isoformat()}.parquet"


def write_days(
    pollutant: Pollutant,
    countries: Sequence[CountryDays],
    days: Iterable[date],
    screening: Screening,
    out: Path,
) -> list[Path]:
    """One asset per day of `days`: every country's points with a value that day, in the
    validated assets' schema. Points the metadata do not describe are left out and logged."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for day in days:
        moment = np.datetime64(day, "D")
        rows: list[PointDays] = []
        described: list[dict[str, Any]] = []
        unknown = 0
        for part in countries:
            for found in sorted(part.found, key=lambda p: p.sampling_point):
                at = found.days == moment
                if not at.any():
                    continue
                point = part.points.get(found.sampling_point)
                if point is None:
                    unknown += 1
                    continue
                rows.append(
                    PointDays(
                        found.sampling_point, found.days[at], found.means[at], found.hours[at]
                    )
                )
                described.append(asdict(point) | {"country": part.country})
        if unknown:
            logger.warning(
                "%s %s: %d points left out, not in the metadata", pollutant, day, unknown
            )
        metadata = {
            "source": "eea-e2a",
            "country": "",
            "countries": sorted({str(p["country"]) for p in described}),
            "pollutant": str(pollutant),
            "unit": "µg/m³",
            "statistic": pollutant.statistic,
            "min_hours": MIN_HOURS,
            "day": day.isoformat(),
            "built": datetime.now(UTC).isoformat(timespec="seconds"),
            "files": sum(part.files for part in countries),
            "left_out": unknown,
            "screening": {
                "ceiling": screening.ceiling,
                "stuck_hours": STUCK_HOURS,
                "removed": screening.of_day(day.isoformat()),
            },
            "dataset_url": DATASET_URL,
            "attribution": ATTRIBUTION,
            "points": described,
        }
        path = out / asset_name(pollutant, day)
        pq.write_table(to_table(rows, metadata), path, compression="zstd")
        logger.info("%s: %d points", path.name, len(rows))
        paths.append(path)
    return paths


def build(
    pollutant: Pollutant,
    first_day: date,
    last_day: date,
    out: Path,
    *,
    countries: Sequence[str] | None = None,
    read: Read = web.read,
    downloads: int = DOWNLOADS,
    metadata_copy: Path | None = None,
) -> list[Path]:
    """The assets of `pollutant` from `first_day` to `last_day`, over `countries` (default:
    every country the service has), and the screening's counts over the whole files, as
    `screening-<pollutant>.json`. `metadata_copy` keeps the EEA's station metadata for the days
    its server does not answer (`read_station_metadata`)."""
    screening = Screening(pollutant)
    metadata = read_station_metadata(metadata_copy, read=read)
    chosen = countries if countries is not None else list_countries(read=read)
    parts = [
        read_country(c, pollutant, metadata, screening, read=read, downloads=downloads)
        for c in chosen
    ]
    days = [date.fromordinal(o) for o in range(first_day.toordinal(), last_day.toordinal() + 1)]
    paths = write_days(pollutant, parts, days, screening, out)
    report = out / f"screening-{pollutant}.json"
    report.write_text(json.dumps(screening.total() | {"files": sum(p.files for p in parts)}))
    return paths


def _get(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
