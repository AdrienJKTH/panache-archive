"""How far the EEA's live hours are from the validated ones, at the same sampling points (Panache's
D92).

The EEA's up-to-date files hold the current year only, and the validated hours stop at the last
year reported: the service itself never holds both for a day. OpenAQ has kept what it took from the
EEA's live feed (its provider "EEA") in a public archive on AWS, one gzipped CSV per location and
day, so a past year's live hours are there. They are compared with the validated daily values of
the archive's release `downscaling-europe-v1`, at the points within `MATCH_METRES` of an OpenAQ
location, through the same screening and the same daily statistics as the live assets.

OpenAQ's values are those it held when the archive was written, maybe updated after the first hour
in places: the gap measured is a lower bound of the live data's own error.
"""

import csv
import gzip
import io
import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from panache_archive import web
from panache_archive.build import METADATA_KEY
from panache_archive.live_stations import Screening
from panache_archive.stations import Pollutant, Read, statistic_of_hours

ARCHIVE: Final = "https://openaq-data-archive.s3.amazonaws.com"
"""OpenAQ's archive on AWS Open Data: anonymous, no key."""
PARAMETERS: Final[Mapping[str, Pollutant]] = {
    "pm25": Pollutant.PM2_5,
    "pm10": Pollutant.PM10,
    "o3": Pollutant.OZONE,
}
UNIT: Final = "µg/m³"
MATCH_METRES: Final = 100.0
"""An OpenAQ location and an EEA sampling point closer than this are the same station."""
EARTH_RADIUS_METRES: Final = 6_371_000.0
LOG_OFFSET: Final = 1.0
"""µg/m³ added before a logarithm, as D90's observation errors."""
CLASSES: Final = ("rural", "suburban", "urban", "traffic", "industrial")
"""Panache's `assimilation.fields.CLASSES`, in its order."""
DOWNLOADS: Final = 32
_KEY: Final = re.compile(r"<Key>([^<]+)</Key>")

logger = logging.getLogger(__name__)


def station_class(type_: str, area: str) -> str:
    """What a point represents, as Panache's `assimilation.fields.station_class`."""
    if type_ == "traffic":
        return "traffic"
    if type_ == "industrial":
        return "industrial"
    if area.startswith("rural"):
        return "rural"
    return "suburban" if area == "suburban" else "urban"


@dataclass(frozen=True, slots=True)
class LiveSeries:
    """A pollutant's hours at an OpenAQ location."""

    location: int
    pollutant: Pollutant
    lat: float
    lon: float
    hours: NDArray[np.datetime64]
    """UTC starts, ascending, distinct."""
    values: NDArray[np.float64]


def list_days(location: int, year: int, *, read: Read = web.read) -> list[str]:
    """The keys of `location`'s files of `year` in the archive."""
    prefix = f"records/csv.gz/locationid={location}/year={year}/"
    keys: list[str] = []
    token = ""
    while True:
        url = f"{ARCHIVE}/?list-type=2&prefix={prefix}{token}"
        answer = read(url).decode()
        keys.extend(_KEY.findall(answer))
        found = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", answer)
        if found is None:
            return keys
        token = f"&continuation-token={quote(found.group(1), safe='')}"


def read_location(location: int, year: int, *, read: Read = web.read) -> list[LiveSeries]:
    """`location`'s hours of `year`, by pollutant, in µg/m³. OpenAQ stamps an hour by its end,
    with its local offset: each becomes its UTC start. An hour held twice keeps its first value."""
    rows: list[dict[str, str]] = []
    for key in list_days(location, year, read=read):
        text = gzip.decompress(read(f"{ARCHIVE}/{key}")).decode("utf-8-sig")
        rows.extend(csv.DictReader(io.StringIO(text)))
    return series_of_rows(location, rows)


def series_of_rows(location: int, rows: Iterable[Mapping[str, str]]) -> list[LiveSeries]:
    """The rows of OpenAQ's files as one series per pollutant."""
    by_pollutant: dict[Pollutant, list[Mapping[str, str]]] = {}
    for row in rows:
        pollutant = PARAMETERS.get(row["parameter"])
        if pollutant is not None and row["units"] == UNIT and row["value"] not in ("", "NaN"):
            by_pollutant.setdefault(pollutant, []).append(row)
    found = []
    for pollutant, kept in sorted(by_pollutant.items()):
        ends = np.array([_utc(row["datetime"]) for row in kept], dtype="datetime64[s]")
        values = np.array([float(row["value"]) for row in kept])
        hours, first = np.unique(ends - np.timedelta64(1, "h"), return_index=True)
        found.append(
            LiveSeries(
                location=location,
                pollutant=pollutant,
                lat=float(kept[0]["lat"]),
                lon=float(kept[0]["lon"]),
                hours=hours,
                values=values[first],
            )
        )
    return found


def _utc(stamp: str) -> np.datetime64:
    """An ISO moment with its offset ("2025-03-02T02:00:00+01:00", "...Z") as UTC."""
    moment = np.datetime64(stamp[:19], "s")
    zone = stamp[19:]
    if zone in ("", "Z"):
        return moment
    minutes = np.timedelta64(int(zone[1:3]) * 60 + int(zone[4:6]), "m")
    return moment - minutes if zone[0] == "+" else moment + minutes


@dataclass(frozen=True, slots=True)
class Validated:
    """A pollutant's validated daily values at its points, from the release's assets."""

    lat: NDArray[np.float64]
    lon: NDArray[np.float64]
    classes: tuple[str, ...]
    point: NDArray[np.int64]
    days: NDArray[np.datetime64]
    means: NDArray[np.float64]


def read_validated(paths: Sequence[Path]) -> Validated:
    """The release's assets of one pollutant, joined, their points' positions and classes."""
    lats: list[float] = []
    lons: list[float] = []
    classes: list[str] = []
    points, days, means = [], [], []
    for path in paths:
        table = pq.read_table(path)
        described = json.loads((table.schema.metadata or {})[METADATA_KEY])
        by_name = {p["sampling_point"]: p for p in described["points"]}
        column = table["sampling_point"].combine_chunks()
        if not isinstance(column, pa.DictionaryArray):
            raise ValueError(f"{path.name}: sampling points should be dictionary-encoded")
        start = len(lats)
        for name in column.dictionary.to_pylist():
            point = by_name[name]
            lats.append(point["lat"])
            lons.append(point["lon"])
            classes.append(station_class(point["type"], point["area"]))
        points.append(np.asarray(column.indices.to_numpy(zero_copy_only=False), np.int64) + start)
        days.append(np.asarray(table["day"].to_numpy(), "datetime64[D]"))
        means.append(np.asarray(table["mean"].to_numpy(zero_copy_only=False), np.float64))
    return Validated(
        lat=np.array(lats),
        lon=np.array(lons),
        classes=tuple(classes),
        point=np.concatenate(points),
        days=np.concatenate(days),
        means=np.concatenate(means),
    )


def nearest(validated: Validated, lat: float, lon: float) -> int | None:
    """The validated point within `MATCH_METRES` of (`lat`, `lon`), the nearest; `None` if none."""
    if not validated.lat.size:
        return None
    phi, other = np.radians(lat), np.radians(validated.lat)
    dphi = other - phi
    dlambda = np.radians(validated.lon - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi) * np.cos(other) * np.sin(dlambda / 2) ** 2
    metres = 2 * EARTH_RADIUS_METRES * np.arcsin(np.sqrt(a))
    best = int(np.argmin(metres))
    return best if metres[best] < MATCH_METRES else None


@dataclass(frozen=True, slots=True)
class Pairs:
    """Days of a pollutant with a validated value at a matched point, and the live value if any."""

    pollutant: Pollutant
    classes: list[str]
    validated: list[float]
    live: list[float]
    """NaN where the live hours gave no value that day."""


def pair(
    pollutant: Pollutant,
    series: Sequence[LiveSeries],
    validated: Validated,
    screening: Screening,
    year: int,
) -> Pairs:
    """Each matched point's validated days of `year`, with the live value of the same day."""
    found = Pairs(pollutant, [], [], [])
    first = np.datetime64(f"{year}-01-01", "D")
    last = np.datetime64(f"{year}-12-31", "D")
    used: set[int] = set()
    for live in series:
        point = nearest(validated, live.lat, live.lon)
        if point is None or point in used:
            continue
        used.add(point)
        kept = screening(live.hours, live.values)
        days = statistic_of_hours(
            str(live.location), live.hours[kept], live.values[kept], pollutant.statistic
        )
        at = (validated.point == point) & (validated.days >= first) & (validated.days <= last)
        live_by_day = dict(zip(days.days.tolist(), days.means.tolist(), strict=True))
        for day, value in zip(
            validated.days[at].tolist(), validated.means[at].tolist(), strict=True
        ):
            found.classes.append(validated.classes[point])
            found.validated.append(value)
            found.live.append(live_by_day.get(day, np.nan))
    logger.info("%s: %d points matched, %d days", pollutant, len(used), len(found.validated))
    return found


def scores(pairs: Pairs) -> list[dict[str, Any]]:
    """By class, and for every class together: the days, the share missing live, and over the days
    with both, the live values' R² against the validated ones, RMSE, bias (live minus validated)
    and, for particles, the standard deviation of the difference of logarithms."""
    classes = np.array(pairs.classes)
    validated = np.array(pairs.validated)
    live = np.array(pairs.live)
    rows = []
    for name in ("all", *CLASSES):
        chosen = np.ones(classes.size, np.bool_) if name == "all" else classes == name
        if not chosen.any():
            continue
        v, x = validated[chosen], live[chosen]
        both = ~np.isnan(x)
        v, x = v[both], x[both]
        row: dict[str, Any] = {
            "pollutant": str(pairs.pollutant),
            "class": name,
            "days": int(chosen.sum()),
            "missing_live": round(float(1 - both.mean()), 3),
        }
        if v.size > 1:
            error = x - v
            total = float(((v - v.mean()) ** 2).sum())
            row |= {
                "r2": round(1 - float((error**2).sum()) / total, 3) if total else None,
                "rmse": round(float(np.sqrt((error**2).mean())), 2),
                "bias": round(float(error.mean()), 2),
            }
            if pairs.pollutant is not Pollutant.OZONE:
                logs = np.log(np.clip(x, 0, None) + LOG_OFFSET) - np.log(v + LOG_OFFSET)
                row["sd_log"] = round(float(logs.std()), 3)
        rows.append(row)
    return rows


def compare(
    locations: Sequence[int],
    year: int,
    validated: Mapping[Pollutant, Validated],
    *,
    read: Read = web.read,
    downloads: int = DOWNLOADS,
) -> dict[str, Any]:
    """The scores of every pollutant of `validated`, and what the screening removed."""
    with ThreadPoolExecutor(max_workers=downloads) as pool:
        every = [
            s
            for found in pool.map(lambda loc: read_location(loc, year, read=read), locations)
            for s in found
        ]
    result: dict[str, Any] = {
        "year": year,
        "locations": len(locations),
        "scores": [],
        "screening": {},
    }
    for pollutant, points in validated.items():
        screening = Screening(pollutant)
        series = [s for s in every if s.pollutant is pollutant]
        found = pair(pollutant, series, points, screening, year)
        result["scores"].extend(scores(found))
        result["screening"][str(pollutant)] = screening.total()
        result.setdefault("series", {})[str(pollutant)] = len(series)
    return result


def markdown(result: Mapping[str, Any]) -> str:
    """The scores as a Markdown table, for the run's summary."""
    head = (
        "| pollutant | class | days | missing live | R² | RMSE | bias | sd log |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    lines = [
        f"| {r['pollutant']} | {r['class']} | {r['days']} | {r['missing_live']:.1%} | "
        f"{r.get('r2')} | {r.get('rmse')} | {r.get('bias')} | {r.get('sd_log', '')} |"
        for r in result["scores"]
    ]
    return head + "\n".join(lines) + "\n\nScreening: " + json.dumps(result["screening"]) + "\n"
