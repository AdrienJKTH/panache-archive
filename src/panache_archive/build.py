"""Build one asset of the archive: a variable's daily statistics over France, month by month.

Each month is downloaded whole from the ADS (the European grid), cut to the box, summed up into
days, and deleted before the next. The asset is a Parquet table, one row per cell and day,
sorted by latitude, longitude and day, so that a place's days lie together.
"""

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from panache_archive import ads
from panache_archive.daily import EVENING_HOURS, daily_max_8h_means, daily_means
from panache_archive.grid import FRANCE, Box, FloatArray
from panache_archive.netcdf import HourlyField, read_field
from panache_archive.variables import (
    ADS_NAME,
    UNIT,
    Reanalysis,
    Statistic,
    Variable,
    statistics,
)

ALL_MONTHS: Final = tuple(range(1, 13))
VALUE_DECIMALS: Final = 2
"""Daily values are rounded to 0.01 μg/m³: far below CAMS's own uncertainty, and the files
shrink by half or more (the decimals of a float32 barely compress)."""
METADATA_KEY: Final = b"panache_archive"
LICENCE: Final = "CC-BY-4.0"
ATTRIBUTION: Final = (
    "Contains modified Copernicus Atmosphere Monitoring Service information [2026]. Neither the "
    "European Commission nor ECMWF is responsible for any use that may be made of the "
    "Copernicus information or data it contains."
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Job:
    """One asset: a variable over consecutive months of a year, from one reanalysis stream."""

    variable: Variable
    year: int
    stream: Reanalysis
    months: tuple[int, ...] = ALL_MONTHS

    def __post_init__(self) -> None:
        first = self.months[0] if self.months else 0
        if not 1 <= first <= 12 or self.months != tuple(range(first, first + len(self.months))):
            raise ValueError(f"months {self.months} are not consecutive months of a year")
        if self.months[-1] > 12:
            raise ValueError(f"months {self.months} run past December")

    @property
    def asset_name(self) -> str:
        """Such as `cams-pm10-2021-validated.parquet`, or `cams-pm10-2021-02-validated.parquet`
        for February alone."""
        first, last = self.months[0], self.months[-1]
        if self.months == ALL_MONTHS:
            span = ""
        elif first == last:
            span = f"-{first:02d}"
        else:
            span = f"-{first:02d}-{last:02d}"
        return f"cams-{self.variable}-{self.year}{span}-{self.stream.label}.parquet"


def build(
    job: Job,
    out_dir: Path,
    workdir: Path,
    *,
    box: Box = FRANCE,
    download: ads.Download = ads.download,
) -> Path:
    """Build `job`'s asset in `out_dir`, downloading into `workdir`; returns its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    columns: dict[Statistic, list[FloatArray]] = {s: [] for s in statistics(job.variable)}
    first: HourlyField | None = None
    evening: FloatArray | None = None
    for month in job.months:
        field = _month(job, month, box, download, workdir)
        if first is None:
            first = field
        elif not (
            np.array_equal(field.lats, first.lats) and np.array_equal(field.lons, first.lons)
        ):
            raise ValueError(f"the grid of {job.year}-{month:02d} differs from the first month's")
        columns[Statistic.MEAN].append(daily_means(field.values))
        if Statistic.MAX_8H_MEAN in columns:
            columns[Statistic.MAX_8H_MEAN].append(daily_max_8h_means(field.values, evening))
            evening = field.values[-EVENING_HOURS:]
    assert first is not None  # `Job` has at least one month

    table = to_table(
        first.lats,
        first.lons,
        date(job.year, job.months[0], 1),
        {statistic: np.concatenate(days) for statistic, days in columns.items()},
        metadata=_metadata(job, box),
    )
    path = out_dir / job.asset_name
    pq.write_table(table, path, compression="zstd")
    logger.info("%s: %d rows, %.1f MB", path.name, table.num_rows, path.stat().st_size / 1e6)
    return path


def to_table(
    lats: FloatArray,
    lons: FloatArray,
    first_day: date,
    columns: Mapping[Statistic, FloatArray],
    *,
    metadata: Mapping[str, Any],
) -> pa.Table:
    """One row per cell and day, sorted by latitude, longitude and day.

    `columns` are `(days, lat, lon)` arrays; NaN becomes null, values are rounded.
    """
    shapes = {values.shape for values in columns.values()}
    if len(shapes) != 1 or next(iter(shapes))[1:] != (lats.size, lons.size):
        raise ValueError(f"columns {shapes} do not match {lats.size} x {lons.size} cells")
    days = next(iter(shapes))[0]
    day_axis = np.datetime64(first_day, "D") + np.arange(days)
    arrays: dict[str, pa.Array[Any]] = {
        "lat": pa.array(np.repeat(lats, lons.size * days)),
        "lon": pa.array(np.tile(np.repeat(lons, days), lats.size)),
        "day": pa.array(np.tile(day_axis, lats.size * lons.size)),
    }
    for statistic, values in columns.items():
        flat = np.round(values.transpose(1, 2, 0).ravel(), VALUE_DECIMALS).astype(np.float32)
        arrays[statistic.value] = pa.array(flat, mask=np.isnan(flat))
    table = pa.table(arrays)
    return table.replace_schema_metadata({METADATA_KEY: json.dumps(metadata).encode()})


def read_metadata(path: Path) -> dict[str, Any]:
    """What an asset says about itself (`_metadata`)."""
    stored = pq.read_schema(path).metadata or {}
    if METADATA_KEY not in stored:
        raise ValueError(f"{path.name} is not an asset of this archive")
    loaded: dict[str, Any] = json.loads(stored[METADATA_KEY])
    return loaded


def _month(job: Job, month: int, box: Box, download: ads.Download, workdir: Path) -> HourlyField:
    """One month's hourly field over the box; its files are deleted once read."""
    started = time.monotonic()
    target = workdir / f"{job.variable}-{job.year}-{month:02d}.download"
    download(ads.request(job.variable, job.year, month, job.stream), target)
    size = target.stat().st_size
    files = ads.netcdf_files(target, workdir)
    try:
        if len(files) != 1:
            raise ValueError(f"expected one netCDF file for {job.year}-{month:02d}: {files}")
        field = read_field(files[0], box)
    finally:
        for path in files:
            path.unlink(missing_ok=True)
    _check_month(field, job.year, month)
    logger.info(
        "%s %d-%02d: %.0f MB downloaded and read in %.0f s",
        job.variable,
        job.year,
        month,
        size / 1e6,
        time.monotonic() - started,
    )
    return field


def _check_month(field: HourlyField, year: int, month: int) -> None:
    """The file is the month asked for, every hour of it, in a mass concentration."""
    start = datetime(year, month, 1, tzinfo=UTC)
    end = datetime(year + month // 12, month % 12 + 1, 1, tzinfo=UTC)
    hours = int((end - start).total_seconds()) // 3600
    if field.first_hour != start or field.hours != hours:
        raise ValueError(
            f"{field.name}: {field.hours} hours from {field.first_hour:%Y-%m-%d %H:%M}, "
            f"expected {hours} from {start:%Y-%m-%d %H:%M}"
        )
    if not field.units.endswith("g/m3"):
        raise ValueError(f"{field.name} is in {field.units!r}, expected μg/m3")


def _metadata(job: Job, box: Box) -> dict[str, Any]:
    return {
        "variable": job.variable.value,
        "unit": UNIT,
        "statistics": [s.value for s in statistics(job.variable)],
        "days": "UTC",
        "decimals": VALUE_DECIMALS,
        "box": asdict(box),
        "dataset": ads.DATASET,
        "doi": ads.DOI,
        "ads_variable": ADS_NAME[job.variable],
        "stream": job.stream.value,
        "model": "ensemble",
        "level": "0",
        "licence": LICENCE,
        "attribution": ATTRIBUTION,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
