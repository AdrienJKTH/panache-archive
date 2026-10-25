"""Build one asset of the archive: a variable's daily statistics over a domain, month by month.

Each month is downloaded whole from the ADS (the European grid), cut to the domain's box, summed
up into days band of latitudes by band, and deleted before the next.
- France's asset is a Parquet table, one row per cell and day, sorted by latitude, longitude and
  day, so that a place's days lie together.
- The whole domain's is a `.cells` file of cell-major frames (`cells`), read one cell at a time.
"""

import json
import logging
import time
from collections.abc import Generator, Mapping
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from panache_archive import ads, cells
from panache_archive.daily import EVENING_HOURS, daily_max_8h_means, daily_means
from panache_archive.domain import Domain
from panache_archive.grid import Box, FloatArray
from panache_archive.netcdf import HourlyField, read_bands
from panache_archive.variables import (
    ADS_NAME,
    UNIT,
    Reanalysis,
    Statistic,
    Variable,
    statistics,
)

ALL_MONTHS: Final = tuple(range(1, 13))
BAND_ROWS: Final = 105
"""Latitudes summed up at a time: France's box in one band, the whole domain in four. A band of
the whole grid's month takes ~0.4 GB in float64, its 8-hour means a few times that."""
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
    """One asset: a variable over consecutive months of a year, from one reanalysis stream, over
    a domain."""

    variable: Variable
    year: int
    stream: Reanalysis
    months: tuple[int, ...] = ALL_MONTHS
    domain: Domain = Domain.FRANCE

    def __post_init__(self) -> None:
        check_months(self.months)

    @property
    def asset_name(self) -> str:
        """Such as `cams-pm10-2021-validated.parquet`, `cams-pm10-2021-02-validated.parquet` for
        February alone, or `cams-pm10-2021-validated.cells` over the whole domain."""
        months = span(self.months)
        return f"cams-{self.variable}-{self.year}{months}-{self.stream.label}{self.domain.suffix}"


def check_months(months: tuple[int, ...]) -> None:
    """Assets cover consecutive months of one year."""
    first = months[0] if months else 0
    if not 1 <= first <= 12 or months != tuple(range(first, first + len(months))):
        raise ValueError(f"months {months} are not consecutive months of a year")
    if months[-1] > 12:
        raise ValueError(f"months {months} run past December")


def span(months: tuple[int, ...]) -> str:
    """How an asset's name tells its months: nothing for a whole year, `-02` for February,
    `-02-04` for February to April."""
    first, last = months[0], months[-1]
    if months == ALL_MONTHS:
        return ""
    return f"-{first:02d}" if first == last else f"-{first:02d}-{last:02d}"


def build(
    job: Job,
    out_dir: Path,
    workdir: Path,
    *,
    box: Box | None = None,
    download: ads.Download = ads.download,
    band_rows: int = BAND_ROWS,
) -> Path:
    """Build `job`'s asset in `out_dir`, downloading into `workdir`; returns its path.

    `box` is the job's domain's unless given."""
    box = job.domain.box if box is None else box
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    wanted = statistics(job.variable)
    columns: dict[Statistic, list[FloatArray]] = {s: [] for s in wanted}
    grid: list[tuple[FloatArray, FloatArray]] = []  # each band's axes, as the first month has them
    evenings: dict[int, FloatArray] = {}  # each band's last hours, for the next month's 8-h means
    for month in job.months:
        days: dict[Statistic, list[FloatArray]] = {s: [] for s in wanted}
        with closing(_month_bands(job, month, box, download, workdir, band_rows)) as bands:
            for index, field in enumerate(bands):
                if index == len(grid):
                    grid.append((field.lats, field.lons))
                elif not (
                    np.array_equal(field.lats, grid[index][0])
                    and np.array_equal(field.lons, grid[index][1])
                ):
                    raise ValueError(
                        f"the grid of {job.year}-{month:02d} differs from the first month's"
                    )
                days[Statistic.MEAN].append(daily_means(field.values))
                if Statistic.MAX_8H_MEAN in days:
                    evening = evenings.get(index)
                    days[Statistic.MAX_8H_MEAN].append(daily_max_8h_means(field.values, evening))
                    evenings[index] = field.values[-EVENING_HOURS:]
        for statistic, by_band in days.items():
            columns[statistic].append(np.concatenate(by_band, axis=1))

    lats = np.concatenate([band_lats for band_lats, _ in grid])
    lons = grid[0][1]
    first_day = date(job.year, job.months[0], 1)
    whole = [np.concatenate(columns.pop(statistic)) for statistic in wanted]  # frees each month
    metadata = _metadata(job, box)
    path = out_dir / job.asset_name
    if job.domain is Domain.FRANCE:
        named = {statistic.value: values for statistic, values in zip(wanted, whole, strict=True)}
        table = to_table(lats, lons, first_day, named, metadata=metadata)
        pq.write_table(table, path, compression="zstd")
    else:
        cells.write(path, lats, lons, first_day, whole, metadata=metadata)
    megabytes = path.stat().st_size / 1e6
    logger.info("%s: %d by %d cells, %.1f MB", path.name, lats.size, lons.size, megabytes)
    return path


def to_table(
    lats: FloatArray,
    lons: FloatArray,
    first_day: date,
    columns: Mapping[str, FloatArray],
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
    for name, values in columns.items():
        flat = np.round(values.transpose(1, 2, 0).ravel(), VALUE_DECIMALS).astype(np.float32)
        arrays[name] = pa.array(flat, mask=np.isnan(flat))
    table = pa.table(arrays)
    return table.replace_schema_metadata({METADATA_KEY: json.dumps(metadata).encode()})


def read_metadata(path: Path) -> dict[str, Any]:
    """What an asset says about itself (`_metadata`)."""
    if path.suffix == Domain.EUROPE.suffix:
        return cells.read_metadata(path)
    stored = pq.read_schema(path).metadata or {}
    if METADATA_KEY not in stored:
        raise ValueError(f"{path.name} is not an asset of this archive")
    loaded: dict[str, Any] = json.loads(stored[METADATA_KEY])
    return loaded


def _month_bands(
    job: Job, month: int, box: Box, download: ads.Download, workdir: Path, band_rows: int
) -> Generator[HourlyField]:
    """One month's hourly field over the box, band by band; its files are deleted once read."""
    started = time.monotonic()
    target = workdir / f"{job.variable}-{job.year}-{month:02d}.download"
    download(ads.request(job.variable, job.year, month, job.stream), target)
    size = target.stat().st_size
    files = ads.netcdf_files(target, workdir)
    try:
        if len(files) != 1:
            raise ValueError(f"expected one netCDF file for {job.year}-{month:02d}: {files}")
        for field in read_bands(files[0], box, band_rows):
            _check_month(field, job.year, month)
            yield field
    finally:
        for path in files:
            path.unlink(missing_ok=True)
    logger.info(
        "%s %d-%02d: %.0f MB downloaded and read in %.0f s",
        job.variable,
        job.year,
        month,
        size / 1e6,
        time.monotonic() - started,
    )


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
        "domain": job.domain.value,
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
