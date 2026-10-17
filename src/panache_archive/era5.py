"""ERA5 daily weather over France, from Open-Meteo's public bucket, as its archive API serves it.

Open-Meteo keeps ERA5 (0.25°, hourly) as `.om` files in its AWS Open Data bucket, read over
HTTPS with no account and no quota: `year_YYYY.om` for past years (until 2021, 2023 for the
boundary layer), then `chunk_N.om` for the 504 hours from N x 504 hours since 1970. Each file is
`[lat 721, lon 1440, hours]`, from -90° and -180° every 0.25°.

The daily columns are the nine of Panache's weather model (`WeatherVariable` in
`panache.sources.weather`), made the way Open-Meteo's archive API makes them from the same
hours (checked in Panache's session 5): UTC days; relative humidity from temperature and dew
point (Magnus); wind speed from each hour's components; the dominant direction from the day's
mean components; pressure in hPa; rain and radiation summed over the 24 hours ending in the day
(MJ/m² for radiation); the boundary layer averaged from its hours.

One difference: the API corrects temperature and dew point for the asked point's altitude, by
the same constant at a place (+1.22 °C at Lyon), and derives humidity from the corrected pair.
The archive keeps the cell's own values, which hold together; a per-place model does not mind a
constant offset.
"""

import asyncio
import json
import logging
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Protocol, Self

import numpy as np
import pyarrow.parquet as pq
from fsspec.implementations.http import HTTPFileSystem
from omfiles import OmFileReaderAsync

from panache_archive.build import ALL_MONTHS, check_months, read_metadata, span, to_table
from panache_archive.check import (
    PLACES,
    Comparison,
    Get,
    Place,
    compare_days,
    floats,
    http_get,
    rounding_step,
    tolerance,
)
from panache_archive.daily import HOURS_PER_DAY, daily_means, daily_sums
from panache_archive.grid import FRANCE, Box, FloatArray, box_cells, nearest

BASE_URL: Final = "https://openmeteo.s3.amazonaws.com/data/copernicus_era5"
ARCHIVE_URL: Final = "https://archive-api.open-meteo.com/v1/archive"
FIRST_YEAR: Final = 2013
"""The archive's first year, as CAMS's."""
CHUNK_HOURS: Final = 504
HUMIDITY_CORRELATION: Final = 0.99
"""The API's humidity comes from altitude-corrected temperature and dew point, which no daily
offset undoes: it is checked by correlation."""
STEP: Final = 0.25
LATS: Final[FloatArray] = np.arange(721, dtype=np.float64) * STEP - 90
LONS: Final[FloatArray] = np.arange(1440, dtype=np.float64) * STEP - 180
EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
DOI: Final = "10.24381/cds.adbb2d47"
LICENCE: Final = "CC-BY-4.0"
ATTRIBUTION: Final = (
    "Contains modified Copernicus Climate Change Service information [2026]. Neither the "
    "European Commission nor ECMWF is responsible for any use that may be made of the "
    "Copernicus information or data it contains. Weather data by Open-Meteo.com (CC-BY 4.0)."
)

logger = logging.getLogger(__name__)


class Hourly(StrEnum):
    """The hourly variables read from the bucket, named as its folders."""

    TEMPERATURE = "temperature_2m"
    """°C."""
    DEW_POINT = "dew_point_2m"
    """°C."""
    WIND_U = "wind_u_component_10m"
    """m/s."""
    WIND_V = "wind_v_component_10m"
    """m/s."""
    PRESSURE = "pressure_msl"
    """Pa."""
    PRECIPITATION = "precipitation"
    """mm in the hour."""
    RADIATION = "shortwave_radiation"
    """W/m², the hour's mean."""
    CLOUD_COVER = "cloud_cover"
    """%."""
    BOUNDARY_LAYER_HEIGHT = "boundary_layer_height"
    """m."""


class Daily(StrEnum):
    """The daily columns, named as Panache's `WeatherVariable` (Open-Meteo's daily names)."""

    TEMPERATURE = "temperature_2m_mean"
    RELATIVE_HUMIDITY = "relative_humidity_2m_mean"
    WIND_SPEED = "wind_speed_10m_mean"
    WIND_DIRECTION = "wind_direction_10m_dominant"
    PRESSURE = "pressure_msl_mean"
    PRECIPITATION = "precipitation_sum"
    RADIATION = "shortwave_radiation_sum"
    CLOUD_COVER = "cloud_cover_mean"
    BOUNDARY_LAYER_HEIGHT = "boundary_layer_height"
    """The archive API has it hourly only: averaged per UTC day, as Panache does."""


ACCUMULATED: Final = frozenset({Hourly.PRECIPITATION, Hourly.RADIATION})
"""Totals or means over the hour before their time stamp."""

UNITS: Final[Mapping[Daily, str]] = {
    Daily.TEMPERATURE: "°C",
    Daily.RELATIVE_HUMIDITY: "%",
    Daily.WIND_SPEED: "m/s",
    Daily.WIND_DIRECTION: "°",
    Daily.PRESSURE: "hPa",
    Daily.PRECIPITATION: "mm",
    Daily.RADIATION: "MJ/m²",
    Daily.CLOUD_COVER: "%",
    Daily.BOUNDARY_LAYER_HEIGHT: "m",
}


class Bucket(Protocol):
    """Where `.om` files are read from."""

    async def exists(self, path: str) -> bool: ...

    async def read(self, path: str, rows: slice, cols: slice, hours: slice) -> FloatArray:
        """`[lat, lon, hour]` values of the file at `path`, relative to the variables' folder."""
        ...


class _Http(HTTPFileSystem):  # type: ignore[misc]
    """omfiles calls `_cat_file(path=…)` and `_size(path=…)`; HTTPFileSystem names them `url`."""

    async def _cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any
    ) -> bytes:
        data: bytes = await super()._cat_file(path, start=start, end=end, **kwargs)
        return data

    async def _size(self, path: str) -> int | None:
        size: int | None = await super()._size(path)
        return size


class HttpBucket:
    """Open-Meteo's bucket over HTTPS, through fsspec as omfiles reads it."""

    def __init__(self, base_url: str = BASE_URL) -> None:
        self._base_url = base_url
        self._fs = _Http(asynchronous=True)

    async def __aenter__(self) -> Self:
        self._session = await self._fs.set_session()
        return self

    async def __aexit__(
        self,
        _type: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        await self._session.close()

    async def exists(self, path: str) -> bool:
        found: bool = await self._fs._exists(f"{self._base_url}/{path}")
        return found

    async def read(self, path: str, rows: slice, cols: slice, hours: slice) -> FloatArray:
        reader = await OmFileReaderAsync.from_fsspec(self._fs, f"{self._base_url}/{path}")
        try:
            values = await reader.read_array((rows, cols, hours))
        finally:
            reader.close()
        return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Era5Job:
    """One asset: France's daily weather over consecutive months of a year."""

    year: int
    months: tuple[int, ...] = ALL_MONTHS

    def __post_init__(self) -> None:
        check_months(self.months)

    @property
    def asset_name(self) -> str:
        """Such as `era5-2021.parquet`, or `era5-2021-01.parquet` for January alone."""
        return f"era5-{self.year}{span(self.months)}.parquet"

    @property
    def first_day(self) -> date:
        return date(self.year, self.months[0], 1)

    @property
    def days(self) -> int:
        last = self.months[-1]
        end = date(self.year + last // 12, last % 12 + 1, 1)
        return (end - self.first_day).days


def last_complete_year(today: date) -> int:
    """The latest year ERA5 covers whole: it runs about five days behind."""
    return today.year - 1 if today >= date(today.year, 1, 10) else today.year - 2


def missing_years(years: Sequence[int], built: set[str]) -> list[Era5Job]:
    return [job for year in years if (job := Era5Job(year)).asset_name not in built]


async def hourly(
    bucket: Bucket, variable: Hourly, job: Era5Job, rows: slice, cols: slice
) -> FloatArray:
    """`[hour, lat, lon]` values of `variable` over the job's days.

    ERA5's accumulations (`ACCUMULATED`) end at their time stamp: the rain stamped 00:00 fell
    from 23:00 the day before. So a day's total takes the 24 hours ending in it, 01:00 to 24:00,
    as the archive API sums them; the other variables take 00:00 to 23:00.
    """
    first = _hour(job.first_day) + (1 if variable in ACCUMULATED else 0)
    return await hours(bucket, variable, first, first + job.days * HOURS_PER_DAY, rows, cols)


async def hours(
    bucket: Bucket, variable: Hourly, first: int, end: int, rows: slice, cols: slice
) -> FloatArray:
    """`[hour, lat, lon]` values of `variable` for the hours `[first, end)` since 1970: from a
    year's file where the bucket has one, else from the 504-hour chunks."""
    parts: list[FloatArray] = []
    start = first
    while start < end:
        year = (EPOCH + timedelta(hours=start)).year
        stop = min(end, _hour(date(year + 1, 1, 1)))
        parts.append(await _year_part(bucket, variable, year, start, stop, rows, cols))
        start = stop
    values = np.concatenate(parts, axis=2)
    if values.shape[2] != end - first:
        raise ValueError(f"{variable}: {values.shape[2]} hours, expected {end - first}")
    return np.transpose(values, (2, 0, 1))


async def _year_part(
    bucket: Bucket,
    variable: Hourly,
    year: int,
    first: int,
    end: int,
    rows: slice,
    cols: slice,
) -> FloatArray:
    """Hours `[first, end)` within one year, as `[lat, lon, hour]`."""
    year_file = f"{variable}/year_{year}.om"
    if await bucket.exists(year_file):
        start = _hour(date(year, 1, 1))
        return await bucket.read(year_file, rows, cols, slice(first - start, end - start))
    reads = [
        bucket.read(
            f"{variable}/chunk_{n}.om",
            rows,
            cols,
            slice(
                max(first, n * CHUNK_HOURS) - n * CHUNK_HOURS,
                min(end, (n + 1) * CHUNK_HOURS) - n * CHUNK_HOURS,
            ),
        )
        for n in range(first // CHUNK_HOURS, (end - 1) // CHUNK_HOURS + 1)
    ]
    return np.concatenate(await asyncio.gather(*reads), axis=2)


def relative_humidity(temperature: FloatArray, dew_point: FloatArray) -> FloatArray:
    """In %, from °C, by the Magnus formula Open-Meteo uses (constants 17.625 and 243.04)."""
    ratio = np.exp(17.625 * dew_point / (243.04 + dew_point)) / np.exp(
        17.625 * temperature / (243.04 + temperature)
    )
    return np.asarray(np.clip(100 * ratio, 0, 100), dtype=np.float64)


def wind_direction(u: FloatArray, v: FloatArray) -> FloatArray:
    """Where the wind blows from, in degrees clockwise from north."""
    return np.asarray(np.degrees(np.arctan2(-u, -v)) % 360, dtype=np.float64)


def daily_weather(values: Mapping[Hourly, FloatArray]) -> dict[Daily, FloatArray]:
    """The nine daily columns from `[hour, lat, lon]` hourly values."""
    u, v = values[Hourly.WIND_U], values[Hourly.WIND_V]
    humidity = relative_humidity(values[Hourly.TEMPERATURE], values[Hourly.DEW_POINT])
    return {
        Daily.TEMPERATURE: daily_means(values[Hourly.TEMPERATURE]),
        Daily.RELATIVE_HUMIDITY: daily_means(humidity),
        Daily.WIND_SPEED: daily_means(np.hypot(u, v)),
        Daily.WIND_DIRECTION: wind_direction(daily_means(u), daily_means(v)),
        Daily.PRESSURE: daily_means(values[Hourly.PRESSURE]) / 100,
        Daily.PRECIPITATION: daily_sums(values[Hourly.PRECIPITATION]),
        Daily.RADIATION: daily_sums(values[Hourly.RADIATION]) * 3600 / 1e6,
        Daily.CLOUD_COVER: daily_means(values[Hourly.CLOUD_COVER]),
        Daily.BOUNDARY_LAYER_HEIGHT: daily_means(values[Hourly.BOUNDARY_LAYER_HEIGHT]),
    }


async def build(job: Era5Job, out_dir: Path, bucket: Bucket, *, box: Box = FRANCE) -> Path:
    """Build `job`'s asset in `out_dir`; returns its path."""
    half = STEP / 2
    cells = box_cells(
        LATS, LONS, Box(box.south - half, box.north + half, box.west - half, box.east + half)
    )
    rows = slice(int(cells.lat_index[0]), int(cells.lat_index[-1]) + 1)
    cols = slice(int(cells.lon_index[0]), int(cells.lon_index[-1]) + 1)
    started = time.monotonic()
    read = await asyncio.gather(*(hourly(bucket, v, job, rows, cols) for v in Hourly))
    logger.info("era5 %s: read in %.0f s", job.asset_name, time.monotonic() - started)
    columns = daily_weather(dict(zip(Hourly, read, strict=True)))

    table = to_table(
        cells.lats,
        cells.lons,
        job.first_day,
        {column.value: values for column, values in columns.items()},
        metadata=_metadata(box),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / job.asset_name
    pq.write_table(table, path, compression="zstd")
    logger.info("%s: %d rows, %.1f MB", path.name, table.num_rows, path.stat().st_size / 1e6)
    return path


def compare(asset: Path, places: Sequence[Place] = PLACES, get: Get = http_get) -> list[Comparison]:
    """The asset against Open-Meteo's archive API at each place, over the asset's days."""
    if read_metadata(asset).get("source") != "era5":
        raise ValueError(f"{asset.name} is not an ERA5 asset")
    table = pq.read_table(asset)
    lats = np.unique(table["lat"].to_numpy())
    lons = np.unique(table["lon"].to_numpy())
    first_day: date = table["day"][0].as_py()
    days = len(np.unique(table["day"].to_numpy()))
    last_day = first_day + timedelta(days=days - 1)

    comparisons: list[Comparison] = []
    for place in places:
        cell = (float(lats[nearest(lats, place.lat)]), float(lons[nearest(lons, place.lon)]))
        mask = (table["lat"].to_numpy() == cell[0]) & (table["lon"].to_numpy() == cell[1])
        rows = table.filter(mask).sort_by("day")
        api = archive_api(place, first_day, last_day, get)
        for column in Daily:
            if column is Daily.BOUNDARY_LAYER_HEIGHT:
                hours = api[column]
                reference, step = daily_means(hours), rounding_step(hours)
            else:
                reference = api[column]
                step = rounding_step(reference)
            allowed = tolerance(step)
            if column is Daily.TEMPERATURE:
                # The offset removed is itself a median of rounded values: up to half a step more.
                allowed += step / 2
            comparisons.append(
                compare_days(
                    place,
                    cell,
                    column.value,
                    floats(rows[column.value]),
                    reference,
                    allowed,
                    offset=column is Daily.TEMPERATURE,
                    circular=column is Daily.WIND_DIRECTION,
                    min_correlation=(
                        HUMIDITY_CORRELATION if column is Daily.RELATIVE_HUMIDITY else None
                    ),
                )
            )
    return comparisons


def archive_api(
    place: Place, first_day: date, last_day: date, get: Get = http_get
) -> dict[Daily, FloatArray]:
    """The archive API's ERA5 days at `place`, as Panache asks for them; the boundary layer is
    hourly."""
    daily = [c.value for c in Daily if c is not Daily.BOUNDARY_LAYER_HEIGHT]
    query = urllib.parse.urlencode(
        {
            "latitude": place.lat,
            "longitude": place.lon,
            "start_date": first_day.isoformat(),
            "end_date": last_day.isoformat(),
            "models": "era5",
            "wind_speed_unit": "ms",
            "timezone": "GMT",
            "daily": ",".join(daily),
            "hourly": Daily.BOUNDARY_LAYER_HEIGHT.value,
        }
    )
    answer = json.loads(get(f"{ARCHIVE_URL}?{query}"))
    days = (last_day - first_day).days + 1
    series: dict[Daily, FloatArray] = {}
    for column in Daily:
        hourly_column = column is Daily.BOUNDARY_LAYER_HEIGHT
        block, units = ("hourly", "hourly_units") if hourly_column else ("daily", "daily_units")
        if answer[units][column.value] != UNITS[column]:
            raise ValueError(f"{column} is in {answer[units][column.value]}, not {UNITS[column]}")
        values = answer[block][column.value]
        expected = days * HOURS_PER_DAY if hourly_column else days
        if len(values) != expected:
            raise ValueError(f"Open-Meteo answered {len(values)} values of {column}")
        series[column] = np.array(
            [np.nan if v is None else float(v) for v in values], dtype=np.float64
        )
    return series


def _hour(day: date) -> int:
    """Hours from 1970 to the start of `day`, UTC."""
    return (datetime(day.year, day.month, day.day, tzinfo=UTC) - EPOCH) // timedelta(hours=1)


def _metadata(box: Box) -> dict[str, Any]:
    return {
        "source": "era5",
        "columns": [c.value for c in Daily],
        "units": {c.value: unit for c, unit in UNITS.items()},
        "days": "UTC",
        "grid": "0.25°",
        "box": asdict(box),
        "bucket": BASE_URL,
        "doi": DOI,
        "temperature": "the grid's, not corrected for altitude as the archive API does",
        "licence": LICENCE,
        "attribution": ATTRIBUTION,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
