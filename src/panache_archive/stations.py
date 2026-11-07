"""Daily values at a country's monitoring stations, from the EEA's validated hours (E1a): the
measurements Panache's downscaling learns from (Panache's D74).

The EEA's Air Quality Download Service lists, for a country, a pollutant and a period, one
Parquet file per sampling point, each holding every validated hour since 2013 (~1.5 MB). Its list
depends on the days asked (Panache's D70), so each year is listed and the lists are joined.

Each file's valid hours are kept, and each UTC day becomes the statistic the European Air Quality
Index classes the pollutant by (Panache's D43), by the rules the archive's CAMS days follow
(`daily`, Panache's D69): the daily mean, from at least 18 valid hours; for ozone, the day's
maximum 8-hour mean, from at least 18 valid 8-hour means of at least 6 hours each. The files'
hours start at UTC+1, whatever the station's own time zone (Panache's D70).

Many networks report their particles as one value a day, weighed on a filter: all of Switzerland's
and nearly all of Ireland's, most of Italy's (Panache's D79). A daily mean needs no hours, so
those values fill the days without enough hours. They are the days the files name, not shifted to
UTC: a day's value spans the country's own day, ~1 hour off a UTC day. Ozone's maximum 8-hour mean
cannot come from them.

The EEA's station metadata give each sampling point its station, position, altitude, type
(traffic, industrial, background) and area (urban, suburban, rural...). They are stored in the
asset's metadata, one entry per sampling point of the table.
"""

import csv
import io
import json
import logging
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from panache_archive import web
from panache_archive.build import METADATA_KEY
from panache_archive.daily import HOURS_PER_DAY, daily_max_8h_means

DOWNLOAD_API: Final = "https://eeadmz1-downloads-api-appservice.azurewebsites.net"
METADATA_URL: Final = "https://discomap.eea.europa.eu/map/fme/metadata/PanEuropean_metadata.csv"
DATASET_URL: Final = (
    "https://www.eea.europa.eu/en/datahub/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d"
)
"""The EEA's page of its Air Quality e-Reporting data, with their re-use terms."""
ATTRIBUTION: Final = (
    "European Environment Agency, Air Quality e-Reporting (validated data, E1a): measurements "
    "of the national networks reported under Directive 2008/50/EC."
)
VALIDATED: Final = 2
"""The service's number for the validated series (E1a); 1 is the up-to-date one, 3 AirBase."""
FIRST_YEAR: Final = 2013
"""The archive's first year, as CAMS's."""
HOURLY: Final = "hour"
DAILY: Final = "day"
"""The service's and the files' names of an hourly and a daily value."""
UNIT: Final = "ug.m-3"
MIN_HOURS: Final = 18
"""Valid hours a UTC day needs for a daily mean (75 %), as CAMS's daily means (Panache's D69)."""
FILE_HOURS_AHEAD_OF_UTC: Final = np.timedelta64(1, "h")
"""Every file's hours start at UTC+1 (Panache's D70)."""
DOWNLOADS: Final = 8
USER_AGENT: Final = "panache-archive (+https://github.com/AdrienJKTH/panache-archive)"
"""discomap.eea.europa.eu answers 403 to urllib's own User-Agent."""
"""Files downloaded at once."""
VALUE_DECIMALS: Final = 2

logger = logging.getLogger(__name__)

type Read = Callable[[urllib.request.Request | str], bytes]


class Pollutant(StrEnum):
    """The pollutants downscaled, named as Panache's CAMS variables: the European Air Quality
    Index's five."""

    PM2_5 = "pm2_5"
    NITROGEN_DIOXIDE = "nitrogen_dioxide"
    PM10 = "pm10"
    OZONE = "ozone"
    SULPHUR_DIOXIDE = "sulphur_dioxide"

    @property
    def notation(self) -> str:
        """The download service's name of the pollutant."""
        return _NOTATIONS[self]

    @property
    def code(self) -> str:
        """The EEA's pollutant code, the last part of `AirPollutantCode` in the metadata."""
        return _CODES[self]

    @property
    def statistic(self) -> str:
        """The daily statistic, named as the archive's CAMS columns: the index's (D43)."""
        return MAX_8H_MEAN if self is Pollutant.OZONE else MEAN


MEAN: Final = "mean"
MAX_8H_MEAN: Final = "max_8h_mean"
_NOTATIONS: Final[Mapping[Pollutant, str]] = {
    Pollutant.PM2_5: "PM2.5",
    Pollutant.NITROGEN_DIOXIDE: "NO2",
    Pollutant.PM10: "PM10",
    Pollutant.OZONE: "O3",
    Pollutant.SULPHUR_DIOXIDE: "SO2",
}
_CODES: Final[Mapping[Pollutant, str]] = {
    Pollutant.PM2_5: "6001",
    Pollutant.NITROGEN_DIOXIDE: "8",
    Pollutant.PM10: "5",
    Pollutant.OZONE: "7",
    Pollutant.SULPHUR_DIOXIDE: "1",
}


@dataclass(frozen=True, slots=True)
class Point:
    """A sampling point, as the EEA's metadata describe it."""

    sampling_point: str
    """E.g. SPO-FR01011_6001."""
    station: str
    """The station's EoI code, e.g. FR01011."""
    lat: float
    lon: float
    altitude: float | None
    """Metres above sea level; `None` when not reported."""
    type: str
    """traffic, industrial or background."""
    area: str
    """urban, suburban, rural, rural-nearcity, rural-regional or rural-remote."""


@dataclass(frozen=True, slots=True)
class PointDays:
    """A sampling point's daily values."""

    sampling_point: str
    days: NDArray[np.datetime64]
    """UTC days, ascending."""
    means: NDArray[np.float64]
    """The day's statistic, µg/m³: its mean, or ozone's maximum 8-hour mean."""
    hours: NDArray[np.int64]
    """Valid hours of the UTC day, 18 to 24 for a mean; 0 for a value reported for the day."""


def list_files(
    country: str, pollutant: Pollutant, years: Iterable[int], *, read: Read = web.read
) -> list[str]:
    """The files the service lists for `country` and `pollutant`, over each of `years`: those of
    hourly values, and of daily ones for a pollutant whose statistic they can give."""
    found: set[str] = set()
    aggregations = (HOURLY,) if pollutant.statistic == MAX_8H_MEAN else (HOURLY, DAILY)
    for year in years:
        for aggregation in aggregations:
            body = {
                "countries": [country],
                "cities": [],
                "pollutants": [pollutant.notation],
                "dataset": VALIDATED,
                "source": "API",
                "dateTimeStart": f"{year}-01-01T00:00:00Z",
                "dateTimeEnd": f"{year + 1}-01-01T00:00:00Z",
                "aggregationType": aggregation,
            }
            request = urllib.request.Request(
                f"{DOWNLOAD_API}/ParquetFile/urls",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            # A CSV of one column, "ParquetFileUrl", behind a byte order mark.
            lines = read(request).decode("utf-8-sig").splitlines()
            if not lines or lines[0].strip() != "ParquetFileUrl":
                raise ValueError(f"unexpected list of files from the EEA for {year}")
            found.update(line.strip() for line in lines[1:] if line.strip())
        logger.info("%s %s %d: %d files listed so far", country, pollutant, year, len(found))
    return sorted(found)


def daily_means(content: bytes, statistic: str = MEAN) -> PointDays:
    """A file's daily values: each UTC day's mean, from at least `MIN_HOURS` valid hours, or its
    maximum 8-hour mean (`MAX_8H_MEAN`), from enough valid 8-hour means (`daily`). A mean's days
    without enough hours take the value the file reports for the day, if any.

    An hour the file holds twice counts once, with its first value.
    """
    table = pq.read_table(
        io.BytesIO(content),
        columns=["Samplingpoint", "Start", "Value", "Unit", "AggType", "Validity"],
    )
    names = set(table["Samplingpoint"].unique().to_pylist())
    if len(names) != 1:
        raise ValueError(f"a file should hold one sampling point, not {sorted(map(str, names))}")
    # "FR/SPO-FR01011_6001": the namespace, then the point as the metadata name it.
    sampling_point = str(names.pop()).rsplit("/", 1)[-1]
    starts = table["Start"].cast(pa.timestamp("s")).to_numpy() - FILE_HOURS_AHEAD_OF_UTC
    values = table["Value"].cast(pa.float64()).to_numpy(zero_copy_only=False)
    values = np.asarray(values, dtype=np.float64)
    aggregation = table["AggType"].to_numpy(zero_copy_only=False)
    valid = (
        (table["Unit"].to_numpy(zero_copy_only=False) == UNIT)
        & (table["Validity"].to_numpy() >= 1)
        & ~np.isnan(values)
    )
    keep = valid & (aggregation == HOURLY)
    hours, first = np.unique(starts[keep], return_index=True)
    hourly = values[keep][first]
    days, index, counts = np.unique(
        hours.astype("datetime64[D]"), return_inverse=True, return_counts=True
    )
    if statistic == MAX_8H_MEAN:
        return _max_8h_means(sampling_point, hours, hourly, days, counts)
    if statistic != MEAN:
        raise ValueError(f"no daily statistic {statistic!r}")
    sums = np.bincount(index, weights=hourly, minlength=days.size)
    full = counts >= MIN_HOURS
    from_hours = PointDays(
        sampling_point=sampling_point,
        days=days[full],
        means=sums[full] / counts[full],
        hours=counts[full].astype(np.int64),
    )
    reported = valid & (aggregation == DAILY)
    named = (starts[reported] + FILE_HOURS_AHEAD_OF_UTC).astype("datetime64[D]")
    return _with_reported_days(from_hours, named, values[reported])


def _with_reported_days(
    from_hours: PointDays, days: NDArray[np.datetime64], values: NDArray[np.float64]
) -> PointDays:
    """`from_hours`, and the days it lacks from values reported for the whole day."""
    named, first = np.unique(days, return_index=True)
    missing = ~np.isin(named, from_hours.days)
    every_day = np.concatenate([from_hours.days, named[missing]])
    order = np.argsort(every_day)
    return PointDays(
        sampling_point=from_hours.sampling_point,
        days=every_day[order],
        means=np.concatenate([from_hours.means, values[first][missing]])[order],
        hours=np.concatenate([from_hours.hours, np.zeros(missing.sum(), np.int64)])[order],
    )


def _max_8h_means(
    sampling_point: str,
    hours: NDArray[np.datetime64],
    hourly: NDArray[np.float64],
    days: NDArray[np.datetime64],
    counts: NDArray[np.int64],
) -> PointDays:
    """Each day's maximum 8-hour mean, from the hours laid on a whole-day UTC series: the day
    before the first one has no hours, so a gap before it reads as missing hours, never as 0."""
    if not days.size:
        return PointDays(sampling_point, days, np.array([]), np.array([], np.int64))
    first = days[0].astype("datetime64[h]")
    series = np.full(int((days[-1] - days[0]).astype(int) + 1) * HOURS_PER_DAY, np.nan)
    series[(hours.astype("datetime64[h]") - first).astype(np.int64)] = hourly
    maxima = daily_max_8h_means(series[:, np.newaxis])[:, 0]
    every_day = days[0] + np.arange(maxima.size)
    found = ~np.isnan(maxima)
    hours_by_day = np.zeros(maxima.size, np.int64)
    hours_by_day[(days - days[0]).astype(np.int64)] = counts
    return PointDays(
        sampling_point=sampling_point,
        days=every_day[found],
        means=maxima[found],
        hours=hours_by_day[found],
    )


def read_points(content: bytes, country: str, pollutant: Pollutant) -> dict[str, Point]:
    """The sampling points of `country` measuring `pollutant` in the EEA's metadata (a CSV of
    tab-separated values), by name. The metadata hold a point once per process and period: its
    first line is kept."""
    points: dict[str, Point] = {}
    rows = csv.DictReader(io.StringIO(content.decode("utf-8-sig")), delimiter="\t")
    for row in rows:
        if row["Countrycode"] != country:
            continue
        if row["AirPollutantCode"].rsplit("/", 1)[-1] != pollutant.code:
            continue
        name = row["SamplingPoint"]
        if name in points:
            continue
        try:
            lat, lon = float(row["Latitude"]), float(row["Longitude"])
        except ValueError:
            continue
        altitude = _number(row["Altitude"])
        points[name] = Point(
            sampling_point=name,
            station=row["AirQualityStationEoICode"],
            lat=lat,
            lon=lon,
            altitude=altitude,
            type=row["AirQualityStationType"],
            area=row["AirQualityStationArea"],
        )
    return points


def asset_name(country: str, pollutant: Pollutant) -> str:
    return f"stations-{country.lower()}-{pollutant}.parquet"


def build(
    country: str,
    pollutant: Pollutant,
    out: Path,
    *,
    last_year: int,
    read: Read = web.read,
    downloads: int = DOWNLOADS,
) -> Path:
    """Every daily value of `pollutant` at `country`'s stations from 2013 to `last_year`, as
    one Parquet asset: one row per sampling point and day, the points in its metadata. The column
    `mean` holds the pollutant's statistic, named in the metadata.

    Points the metadata do not describe are left out, since a mean without a position is of no
    use to a map; they are logged.
    """
    years = range(FIRST_YEAR, last_year + 1)
    urls = list_files(country, pollutant, years, read=read)
    points = read_points(read(_get(METADATA_URL)), country, pollutant)
    with ThreadPoolExecutor(max_workers=downloads) as pool:
        found = list(pool.map(lambda url: daily_means(read(_get(url)), pollutant.statistic), urls))
    last_day = np.datetime64(date(last_year, 12, 31), "D")
    kept = [_until(p, last_day) for p in found]
    unknown = sorted(p.sampling_point for p in kept if p.sampling_point not in points)
    if unknown:
        logger.warning("%d points left out, not in the metadata: %s", len(unknown), unknown)
    described = sorted(
        (p for p in kept if p.sampling_point in points and p.days.size),
        key=lambda p: p.sampling_point,
    )
    metadata = {
        "source": "eea-e1a",
        "country": country,
        "pollutant": str(pollutant),
        "unit": "µg/m³",
        "statistic": pollutant.statistic,
        "min_hours": MIN_HOURS,
        "first_year": FIRST_YEAR,
        "last_year": last_year,
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": len(urls),
        "dataset_url": DATASET_URL,
        "attribution": ATTRIBUTION,
        "points": [asdict(points[p.sampling_point]) for p in described],
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / asset_name(country, pollutant)
    pq.write_table(to_table(described, metadata), path, compression="zstd")
    rows = sum(p.days.size for p in described)
    logger.info("%s: %d points, %d daily means", path.name, len(described), rows)
    return path


def to_table(found: Sequence[PointDays], metadata: Mapping[str, Any]) -> pa.Table:
    """One row per sampling point and day, in the order given."""
    names = [p.sampling_point for p in found]
    lengths = [p.days.size for p in found]
    point = pa.DictionaryArray.from_arrays(
        np.repeat(np.arange(len(found), dtype=np.int32), lengths), pa.array(names)
    )
    days = np.concatenate([p.days for p in found]) if found else np.array([], "datetime64[D]")
    means = np.concatenate([p.means for p in found]) if found else np.array([])
    hours = np.concatenate([p.hours for p in found]) if found else np.array([], np.int64)
    table = pa.table(
        {
            "sampling_point": point,
            "day": pa.array(days.astype("datetime64[D]"), pa.date32()),
            "mean": pa.array(np.round(means, VALUE_DECIMALS).astype(np.float32)),
            "hours": pa.array(hours.astype(np.int8)),
        }
    )
    return table.replace_schema_metadata({METADATA_KEY: json.dumps(metadata).encode()})


def _get(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def _until(found: PointDays, last_day: np.datetime64) -> PointDays:
    """`found`'s days up to `last_day`: the files hold the current year's first validated hours."""
    keep = found.days <= last_day
    return PointDays(found.sampling_point, found.days[keep], found.means[keep], found.hours[keep])


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return None if value == -9999 else value
