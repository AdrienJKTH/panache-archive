"""Check an asset against what Open-Meteo serves at a few places.

Open-Meteo serves the same CAMS reanalysis hour by hour, and Panache sums its days by the same
rules, so the two must agree. Open-Meteo rounds hourly values to 0.1 μg/m³ and the archive daily
ones to 0.01, hence a tolerance of 0.1 μg/m³.
"""

from __future__ import annotations  # pyarrow's arrays are generic in their stubs only

import json
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from panache_archive.build import read_metadata
from panache_archive.daily import HOURS_PER_DAY, daily_max_8h_means, daily_means
from panache_archive.grid import FloatArray, nearest
from panache_archive.variables import Statistic, Variable, statistics

OPEN_METEO_URL: Final = "https://air-quality-api.open-meteo.com/v1/air-quality"
TIMEOUT_SECONDS: Final = 60
TOLERANCE: Final = 0.1
"""Largest daily difference, in μg/m³, for an asset to agree with Open-Meteo."""

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


@dataclass(frozen=True, slots=True)
class Comparison:
    """One place and statistic: the archive's days against Open-Meteo's."""

    place: Place
    cell: tuple[float, float]
    statistic: Statistic
    days: int
    """Days with a value on both sides."""
    max_difference: float
    correlation: float
    archive_peak: float
    open_meteo_peak: float

    @property
    def agrees(self) -> bool:
        return self.days > 0 and self.max_difference <= TOLERANCE


def http_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as answer:
        return bytes(answer.read())


def compare(asset: Path, places: Sequence[Place] = PLACES, get: Get = http_get) -> list[Comparison]:
    """The asset against Open-Meteo at each place, over the asset's days."""
    variable = Variable(read_metadata(asset)["variable"])
    table = pq.read_table(asset)
    lats = np.unique(table["lat"].to_numpy())
    lons = np.unique(table["lon"].to_numpy())
    first_day: date = pc.min(table["day"]).as_py()
    last_day: date = pc.max(table["day"]).as_py()

    comparisons: list[Comparison] = []
    for place in places:
        cell = (float(lats[nearest(lats, place.lat)]), float(lons[nearest(lons, place.lon)]))
        at_cell = pc.and_(
            pc.equal(table["lat"], pa.scalar(cell[0])), pc.equal(table["lon"], pa.scalar(cell[1]))
        )
        rows = table.filter(at_cell).sort_by("day")
        hourly = open_meteo_hourly(variable, place, first_day, last_day, get)
        for statistic in statistics(variable):
            reference = (
                daily_means(hourly) if statistic is Statistic.MEAN else daily_max_8h_means(hourly)
            )
            archived = _floats(rows[statistic.value])
            comparisons.append(_compare(place, cell, statistic, archived, reference))
    return comparisons


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


def render(asset_name: str, comparisons: Sequence[Comparison]) -> str:
    """A Markdown table, for a run's summary."""
    lines = [
        f"### `{asset_name}` against Open-Meteo",
        "",
        "| Place | Cell | Statistic | Days | Max difference | r | Peak, archive / Open-Meteo | |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in comparisons:
        lines.append(
            f"| {c.place.name} | {c.cell[0]:.2f}, {c.cell[1]:.2f} | {c.statistic.value} "
            f"| {c.days} | {c.max_difference:.2f} | {c.correlation:.3f} "
            f"| {c.archive_peak:.1f} / {c.open_meteo_peak:.1f} | {'✓' if c.agrees else '✗'} |"
        )
    return "\n".join(lines) + "\n"


def _floats(column: pa.ChunkedArray[Any]) -> FloatArray:
    """A float column, nulls as NaN."""
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)


def _compare(
    place: Place,
    cell: tuple[float, float],
    statistic: Statistic,
    archived: FloatArray,
    reference: FloatArray,
) -> Comparison:
    if archived.shape != reference.shape:
        raise ValueError(f"{archived.size} archived days for {reference.size} from Open-Meteo")
    both = ~np.isnan(archived) & ~np.isnan(reference)
    a, r = archived[both], reference[both]
    days = int(both.sum())
    spread = days > 1 and float(np.std(a)) > 0 and float(np.std(r)) > 0
    return Comparison(
        place=place,
        cell=cell,
        statistic=statistic,
        days=days,
        max_difference=float(np.max(np.abs(a - r))) if days else float("nan"),
        correlation=float(np.corrcoef(a, r)[0, 1]) if spread else float("nan"),
        archive_peak=float(np.max(a)) if days else float("nan"),
        open_meteo_peak=float(np.max(r)) if days else float("nan"),
    )
