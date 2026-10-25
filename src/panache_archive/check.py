"""Check an asset against what Open-Meteo serves at a few places, and the whole domain's against
France's.

Open-Meteo serves the same CAMS reanalysis hour by hour, and Panache sums its days by the same
rules, so the two must agree, but for rounding. Open-Meteo rounds its hourly values, to 0.1 μg/m³
for most variables and to whole μg/m³ for some, such as ozone. A mean of rounded values lies
within half a step of the exact mean, and so does the highest of such means. The archive rounds
its daily values to 0.01. Hence the tolerance: half Open-Meteo's step, plus 0.005 μg/m³.

Open-Meteo serves the reanalysis until 2021 only (Panache's D45). France's assets are built from
the same downloads by the same rules, so a whole-domain asset must hold their very numbers, every
year: its "twin" check.
"""

from __future__ import annotations  # pyarrow's arrays are generic in their stubs only

import json
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from panache_archive import cells, web
from panache_archive.build import VALUE_DECIMALS, read_metadata
from panache_archive.daily import HOURS_PER_DAY, daily_max_8h_means, daily_means
from panache_archive.domain import Domain
from panache_archive.grid import COORDINATE_DECIMALS, FloatArray, nearest
from panache_archive.variables import Statistic, Variable, statistics

OPEN_METEO_URL: Final = "https://air-quality-api.open-meteo.com/v1/air-quality"
STEPS: Final = (1.0, 0.1, 0.01)
"""The rounding steps Open-Meteo may use, coarsest first, in μg/m³."""
ARCHIVE_ROUNDING: Final = 0.5 * 10**-VALUE_DECIMALS

Get = Callable[[str], bytes]
"""Fetches a URL and returns the answer's body."""


@dataclass(frozen=True, slots=True)
class Place:
    name: str
    lat: float
    lon: float


PLACES: Final = (
    Place("Paris", 48.857, 2.352),
    Place("Lyon", 45.76, 4.84),
    Place("Grenoble", 45.19, 5.72),
    Place("Bordeaux", 44.838, -0.579),
)
"""Away from cell edges, so that both pick the same cell: asked at (45.2, 5.7), on a corner,
Grenoble once got a neighbour from one of them."""
EUROPE_PLACES: Final = (
    Place("Lyon", 45.76, 4.84),
    Place("Madrid", 40.425, -3.675),
    Place("Warsaw", 52.225, 21.025),
    Place("Athens", 37.975, 23.725),
    Place("Helsinki", 60.175, 24.925),
)
"""Across the whole domain, a quarter of a cell from the edges of both grids; Lyon ties it to
France's places."""

DaysAt = Callable[[Place], tuple[tuple[float, float], Mapping[str, FloatArray]]]
"""A place's cell, and its days by column."""


@dataclass(frozen=True, slots=True)
class Comparison:
    """One place and column: the archive's days against Open-Meteo's."""

    place: Place
    cell: tuple[float, float]
    column: str
    days: int
    """Days with a value on both sides."""
    max_difference: float
    tolerance: float
    """Largest difference that rounding explains (`tolerance`)."""
    correlation: float
    archive_peak: float
    open_meteo_peak: float
    offset: float = 0.0
    """A constant difference removed before comparing (ERA5 temperature: the API corrects it for
    the asked point's altitude)."""

    min_correlation: float | None = None
    """When set, agreement is judged by correlation instead of by difference."""

    @property
    def agrees(self) -> bool:
        """Within the tolerance, give or take float noise (two values rounded to 0.01 may
        differ by 0.0100001); or correlated enough, when `min_correlation` is set."""
        if self.days == 0:
            return False
        if self.min_correlation is not None:
            return self.correlation >= self.min_correlation
        return self.max_difference <= self.tolerance + 1e-6

    @property
    def criterion(self) -> str:
        """What agreement means here, for the report."""
        if self.min_correlation is not None:
            return f"r ≥ {self.min_correlation:.2f}"
        return f"{self.tolerance:.3f}"


def http_get(url: str) -> bytes:
    return web.read(url)


def compare(
    asset: Path, places: Sequence[Place] | None = None, get: Get = http_get
) -> list[Comparison]:
    """The asset against Open-Meteo at each place (the domain's own places unless given), over
    the asset's days."""
    variable = Variable(read_metadata(asset)["variable"])
    whole_domain = asset.suffix == Domain.EUROPE.suffix
    first_day, last_day, days_at = (_cells_days if whole_domain else _table_days)(asset)
    if places is None:
        places = EUROPE_PLACES if whole_domain else PLACES

    comparisons: list[Comparison] = []
    for place in places:
        cell, columns = days_at(place)
        hourly = open_meteo_hourly(variable, place, first_day, last_day, get)
        allowed = tolerance(rounding_step(hourly))
        for statistic in statistics(variable):
            reference = (
                daily_means(hourly) if statistic is Statistic.MEAN else daily_max_8h_means(hourly)
            )
            comparisons.append(
                compare_days(
                    place, cell, statistic.value, columns[statistic.value], reference, allowed
                )
            )
    return comparisons


def _table_days(asset: Path) -> tuple[date, date, DaysAt]:
    table = pq.read_table(asset)
    lats = np.unique(table["lat"].to_numpy())
    lons = np.unique(table["lon"].to_numpy())
    first_day: date = pc.min(table["day"]).as_py()
    last_day: date = pc.max(table["day"]).as_py()
    names = [n for n in table.column_names if n not in ("lat", "lon", "day")]

    def days_at(place: Place) -> tuple[tuple[float, float], Mapping[str, FloatArray]]:
        cell = (float(lats[nearest(lats, place.lat)]), float(lons[nearest(lons, place.lon)]))
        at_cell = pc.and_(
            pc.equal(table["lat"], pa.scalar(cell[0])), pc.equal(table["lon"], pa.scalar(cell[1]))
        )
        rows = table.filter(at_cell).sort_by("day")
        return cell, {name: floats(rows[name]) for name in names}

    return first_day, last_day, days_at


def _cells_days(asset: Path) -> tuple[date, date, DaysAt]:
    header = cells.read_header(asset)
    names: list[str] = read_metadata(asset)["statistics"]
    lats, lons = header.lats, header.lons

    def days_at(place: Place) -> tuple[tuple[float, float], Mapping[str, FloatArray]]:
        row, column = nearest(lats, place.lat), nearest(lons, place.lon)
        values = cells.to_values(cells.read_cell(asset, row, column), header.scale)
        return (float(lats[row]), float(lons[column])), dict(zip(names, values, strict=True))

    last_day = header.first_day + timedelta(days=header.days - 1)
    return header.first_day, last_day, days_at


@dataclass(frozen=True, slots=True)
class TwinComparison:
    """A whole-domain asset against France's asset of the same variable-year, over France's
    cells and the days both hold."""

    twin: str
    cells: int
    days: int
    differing: int
    """Values that differ, a missing value against a number included."""
    max_difference: float

    @property
    def agrees(self) -> bool:
        return self.cells > 0 and self.days > 0 and self.differing == 0


def compare_twin(asset: Path, twin: Path) -> TwinComparison:
    """`asset` (`.cells`) against `twin` (France's Parquet): the very same float32 numbers."""
    header = cells.read_header(asset)
    names: list[str] = read_metadata(asset)["statistics"]
    table = pq.read_table(twin)
    lats = np.unique(table["lat"].to_numpy())
    lons = np.unique(table["lon"].to_numpy())
    twin_first: date = pc.min(table["day"]).as_py()
    twin_days = table.num_rows // (lats.size * lons.size)
    rows, columns = _within(header.lats, lats), _within(header.lons, lons)
    region = cells.read_region(asset, rows, columns)  # (statistic, day, lat, lon)

    start = max(header.first_day, twin_first)
    end = min(header.first_day + timedelta(days=header.days), twin_first + timedelta(twin_days))
    days = max(0, (end - start).days)
    differing, largest = 0, 0.0
    for index, name in enumerate(names):
        theirs = floats(table[name]).astype(np.float32).reshape(lats.size, lons.size, twin_days)
        theirs = theirs[:, :, (start - twin_first).days :][:, :, :days]
        ours = cells.to_values(region[index], header.scale).astype(np.float32).transpose(1, 2, 0)
        ours = ours[:, :, (start - header.first_day).days :][:, :, :days]
        same = (ours == theirs) | (np.isnan(ours) & np.isnan(theirs))
        differing += int((~same).sum())
        both = ~np.isnan(ours) & ~np.isnan(theirs)
        if both.any():
            largest = max(largest, float(np.max(np.abs(ours[both] - theirs[both]))))
    return TwinComparison(
        twin=twin.name,
        cells=lats.size * lons.size,
        days=days,
        differing=differing,
        max_difference=largest,
    )


def render_twin(asset_name: str, twin: TwinComparison) -> str:
    """One Markdown line, for a run's summary."""
    return (
        f"`{asset_name}` against France's `{twin.twin}`: {twin.cells} cells by {twin.days} days, "
        f"{twin.differing} values differ (largest by {twin.max_difference:.2f}) "
        f"{'✓' if twin.agrees else '✗'}\n"
    )


def _within(axis: FloatArray, values: FloatArray) -> range:
    """The consecutive indices of `values` in `axis`, both ascending cell centres."""
    rounded = np.round(axis, COORDINATE_DECIMALS)
    index = np.searchsorted(rounded, np.round(values, COORDINATE_DECIMALS))
    found = index < axis.size
    found[found] = rounded[index[found]] == np.round(values[found], COORDINATE_DECIMALS)
    if not found.all() or np.any(np.diff(index) != 1):
        raise ValueError("the twin's cells are not a block of the asset's grid")
    return range(int(index[0]), int(index[-1]) + 1)


def open_meteo_hourly(
    variable: Variable, place: Place, first_day: date, last_day: date, get: Get = http_get
) -> FloatArray:
    """Open-Meteo's hourly CAMS Europe values at `place`, whole UTC days; NaN where null."""
    query = urllib.parse.urlencode(
        {
            "latitude": place.lat,
            "longitude": place.lon,
            "hourly": variable.value,
            "domains": "cams_europe",
            "timezone": "GMT",
            "start_date": first_day.isoformat(),
            "end_date": last_day.isoformat(),
        }
    )
    hourly = json.loads(get(f"{OPEN_METEO_URL}?{query}"))["hourly"]
    times: list[str] = hourly["time"]
    hours = ((last_day - first_day) // timedelta(days=1) + 1) * HOURS_PER_DAY
    if not times or times[0] != f"{first_day.isoformat()}T00:00" or len(times) != hours:
        raise ValueError(f"Open-Meteo answered {len(times)} hours, expected {hours}")
    return np.array(
        [np.nan if v is None else float(v) for v in hourly[variable.value]], dtype=np.float64
    )


def rounding_step(values: FloatArray) -> float:
    """How Open-Meteo rounded `values`: the coarsest step that each is a multiple of, else 0."""
    known = values[~np.isnan(values)]
    for step in STEPS:
        scaled = known / step
        if np.allclose(scaled, np.round(scaled), rtol=0, atol=1e-6):
            return step
    return 0.0


def tolerance(step: float) -> float:
    """The largest daily difference that rounding explains: half of Open-Meteo's step, plus the
    archive's own rounding."""
    return step / 2 + ARCHIVE_ROUNDING


def render(asset_name: str, comparisons: Sequence[Comparison]) -> str:
    """A Markdown table, for a run's summary."""
    lines = [
        f"### `{asset_name}` against Open-Meteo",
        "",
        "| Place | Cell | Column | Days | Max difference | Tolerance | r "
        "| Peak, archive / Open-Meteo | |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in comparisons:
        lines.append(
            f"| {c.place.name} | {c.cell[0]:.2f}, {c.cell[1]:.2f} | {c.column} "
            f"| {c.days} | {c.max_difference:.2f}{_offset(c.offset)} | {c.criterion} "
            f"| {c.correlation:.3f} "
            f"| {c.archive_peak:.1f} / {c.open_meteo_peak:.1f} | {'✓' if c.agrees else '✗'} |"
        )
    return "\n".join(lines) + "\n"


def _offset(offset: float) -> str:
    return f" (offset {offset:+.2f})" if offset else ""


def floats(column: pa.ChunkedArray[Any]) -> FloatArray:
    """A float column, nulls as NaN."""
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)


def compare_days(
    place: Place,
    cell: tuple[float, float],
    column: str,
    archived: FloatArray,
    reference: FloatArray,
    allowed: float,
    *,
    offset: bool = False,
    circular: bool = False,
    min_correlation: float | None = None,
) -> Comparison:
    """The archive's days against Open-Meteo's. `offset` removes their median difference first;
    `circular` compares angles in degrees (wind directions), with no correlation;
    `min_correlation` judges by correlation instead of by difference."""
    if archived.shape != reference.shape:
        raise ValueError(f"{archived.size} archived days for {reference.size} from Open-Meteo")
    both = ~np.isnan(archived) & ~np.isnan(reference)
    a, r = archived[both], reference[both]
    days = int(both.sum())
    differences = a - r
    if circular:
        differences = (differences + 180) % 360 - 180
    removed = float(np.median(differences)) if offset and days else 0.0
    spread = not circular and days > 1 and float(np.std(a)) > 0 and float(np.std(r)) > 0
    return Comparison(
        place=place,
        cell=cell,
        column=column,
        days=days,
        max_difference=float(np.max(np.abs(differences - removed))) if days else float("nan"),
        tolerance=allowed,
        correlation=float(np.corrcoef(a, r)[0, 1]) if spread else float("nan"),
        archive_peak=float(np.max(a)) if days else float("nan"),
        open_meteo_peak=float(np.max(r)) if days else float("nan"),
        offset=removed,
        min_correlation=min_correlation,
    )
