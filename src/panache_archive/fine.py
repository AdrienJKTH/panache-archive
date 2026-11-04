"""Fixed layers over Europe at 3 arc seconds (~90 by 65 m): where a point lies within its 1 km
cell, for Panache's downscaling below 1 km (Panache's D74, Adrien's requests of 2026-11-01 and
2026-11-04).

The grid is 1/1200° over the CAMS Europe domain, 30 to 72° N and 25° W to 45° E: ten by ten cells
in each 30″ cell of the 1 km layers and 120 by 120 in each 0.1° CAMS cell. It is built one block
of 10° by 10° at a time (`blocks`), since the whole grid is 4.2 billion cells, and cut into 1°
tiles, one `.npz` each (`fine-N45E004.npz` holds 45-46° N, 4-5° E), rows from the south; a tile
without CORINE's land is left out, so Europe here is what CORINE covers.

- **CORINE Land Cover 2018** (100 m, the Copernicus Land Monitoring Service): each cell's class,
  as the raster's index 1 to 44 (`CORINE_CODES` gives the CLC code); 0 outside CORINE. Its
  objects are at least 25 ha: below ~500 m it tells nothing. Its water classes give the land.
- **Population** (inhabitants per km²): JRC's GHS-POP 2020 at 100 m.
- **Built-up surface** (share of the cell, in 250ths): JRC's GHS-BUILT-S 2020 at 100 m.
- **Elevation** (m): the Copernicus DEM GLO-90.
- **Major roads** (m in the cell): OpenStreetMap's motorways and trunks, primary and secondary
  roads, links included, from Geofabrik's extract of Europe.

GDAL cuts the rasters onto the grid (`scripts/fine-layers.sh`); this module sums the roads' lengths
cell by cell and writes the tiles.
"""

import json
import logging
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from panache_archive.grid import Box
from panache_archive.static import EARTH_RADIUS_KM, Grid, read_layer

FloatArray = NDArray[np.float64]

CELLS_PER_DEGREE: Final = 1200
DOMAIN: Final = Box(south=30.0, north=72.0, west=-25.0, east=45.0)
"""The CAMS Europe domain, on whole degrees."""
BLOCK_DEGREES: Final = 10
EAST_FROM: Final = 15
"""Blocks from this longitude go to the eastern release: a release holds at most 1,000 assets."""
RELEASES: Final = ("fine-europe-west-v1", "fine-europe-east-v1")
TILE_DEGREES: Final = 1
GHSL_RADIUS_M: Final = 6378137.0
"""GHSL's Mollweide (ESRI:54009) is on WGS 84's sphere."""
GHSL_WEST_M: Final = -18041000.0
GHSL_NORTH_M: Final = 9000000.0
GHSL_TILE_M: Final = 1000000.0
"""GHSL's 100 m tiles: 1,000 km squares from this corner, `R1_C1` the first."""
STEP_M: Final = 10.0
"""Roads are cut into steps of at most this length, each counted in the cell of its middle."""
ROAD_CLASSES: Final[Mapping[str, str]] = {
    "motorway": "road_major",
    "motorway_link": "road_major",
    "trunk": "road_major",
    "trunk_link": "road_major",
    "primary": "road_primary",
    "primary_link": "road_primary",
    "secondary": "road_secondary",
    "secondary_link": "road_secondary",
}
ROAD_LAYERS: Final = ("road_major", "road_primary", "road_secondary")
CORINE_CODES: Final = (
    111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142, 211, 212, 213, 221, 222, 223, 231,
    241, 242, 243, 244, 311, 312, 313, 321, 322, 323, 324, 331, 332, 333, 334, 335, 411, 412,
    421, 422, 423, 511, 512, 521, 522, 523,
)  # fmt: skip
"""The CLC code of each raster index, 1 to 44."""
FIRST_WATER_INDEX: Final = 40
"""Indices 40 to 44 are water: courses, bodies, lagoons, estuaries, sea."""
BUILT_UP_SCALE: Final = 250
GHSL_PIXEL_KM2: Final = 0.01
"""A GHSL 100 m pixel's area: GDAL averages its counts (residents, built-up m²) onto the grid."""
SOURCES: Final = {
    "corine": "CORINE Land Cover 2018, raster 100 m, v2020_20u1 (Copernicus Land Monitoring "
    "Service, European Environment Agency)",
    "population": "JRC GHS-POP R2023A, epoch 2020, 100 m (CC-BY 4.0)",
    "built_up": "JRC GHS-BUILT-S R2023A, epoch 2020, 100 m (CC-BY 4.0)",
    "elevation": "Copernicus DEM GLO-90 (© DLR e.V. 2010-2014 and © Airbus Defence and Space "
    "GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA)",
    "roads": "© OpenStreetMap contributors (ODbL), Geofabrik's extract of Europe",
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Road:
    layer: str
    """One of `ROAD_LAYERS`."""
    lines: list[list[tuple[float, float]]]
    """Its lines, each a list of (lon, lat)."""


def read_roads(path: Path) -> Iterator[Road]:
    """The major roads of a GeoJSON text sequence (osmium's `geojsonseq` export), one feature a
    line, possibly behind the record separator."""
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            text = line.strip().lstrip("\x1e")
            if not text:
                continue
            feature = json.loads(text)
            layer = ROAD_CLASSES.get((feature.get("properties") or {}).get("highway", ""))
            geometry = feature.get("geometry") or {}
            if layer is None:
                continue
            if geometry.get("type") == "LineString":
                yield Road(layer, [_points(geometry["coordinates"])])
            elif geometry.get("type") == "MultiLineString":
                yield Road(layer, [_points(part) for part in geometry["coordinates"]])


def road_lengths(roads: Iterable[Road], grid: Grid) -> dict[str, NDArray[np.uint16]]:
    """Each road layer's metres in each cell, `[lat, lon]` from the south, capped at 65,535.

    The steps are gathered first and summed once per layer, so that only one grid of float64
    is ever in memory (~1.2 GB for a block).
    """
    rows, cols = grid.shape
    cells: dict[str, list[NDArray[np.intp]]] = {layer: [] for layer in ROAD_LAYERS}
    metres: dict[str, list[FloatArray]] = {layer: [] for layer in ROAD_LAYERS}
    for road in roads:
        for line in road.lines:
            found, lengths = _steps(np.asarray(line, dtype=np.float64), grid)
            cells[road.layer].append(found)
            metres[road.layer].append(lengths)
    totals: dict[str, NDArray[np.uint16]] = {}
    cap = np.iinfo(np.uint16).max
    for layer in ROAD_LAYERS:
        index = np.concatenate([np.array([], np.intp), *cells.pop(layer)])
        weights = np.concatenate([np.array([]), *metres.pop(layer)])
        summed = np.bincount(index, weights=weights, minlength=rows * cols)
        totals[layer] = summed.round().clip(max=cap).astype(np.uint16).reshape(rows, cols)
    return totals


def tile_name(south: int, west: int) -> str:
    lat = f"{'N' if south >= 0 else 'S'}{abs(south):02d}"
    lon = f"{'E' if west >= 0 else 'W'}{abs(west):03d}"
    return f"fine-{lat}{lon}.npz"


def write_tiles(layers: Mapping[str, NDArray[Any]], out: Path, grid: Grid) -> list[Path]:
    """One `.npz` per whole degree of latitude and longitude holding land, cut at the box's edges;
    `layers` are `[lat, lon]` arrays from the south, `corine` among them."""
    out.mkdir(parents=True, exist_ok=True)
    about = {
        "grid": {"cells_per_degree": grid.cells_per_degree, "rows": "from the south"},
        "corine_codes": list(CORINE_CODES),
        "units": {
            "corine": "CORINE raster index, 1 to 44 (corine_codes), 0 outside CORINE",
            "population": "inhabitants/km²",
            "built_up": f"share of the cell in {BUILT_UP_SCALE}ths",
            "elevation": "m",
            **dict.fromkeys(ROAD_LAYERS, "m of road in the cell"),
        },
        "sources": SOURCES,
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    written: list[Path] = []
    for south in range(math.floor(grid.box.south), math.ceil(grid.box.north)):
        rows = _cells(south, grid.box.south, grid.box.north, grid)
        for west in range(math.floor(grid.box.west), math.ceil(grid.box.east)):
            cols = _cells(west, grid.box.west, grid.box.east, grid)
            corine = layers["corine"][rows, cols]
            if not ((corine >= 1) & (corine < FIRST_WATER_INDEX)).any():
                continue
            tile_about = {
                **about,
                "south": max(south, grid.box.south),
                "west": max(west, grid.box.west),
            }
            arrays: dict[str, Any] = {name: values[rows, cols] for name, values in layers.items()}
            path = out / tile_name(south, west)
            np.savez_compressed(path, about=np.array(json.dumps(tile_about)), **arrays)
            written.append(path)
    return written


def _cells(first: int, low: float, high: float, grid: Grid) -> slice:
    """The grid's cells, along one axis from `low` to `high`, within the degree from `first`."""
    start = round((max(first, low) - low) * grid.cells_per_degree)
    stop = round((min(first + TILE_DEGREES, high) - low) * grid.cells_per_degree)
    return slice(start, stop)


def blocks(domain: Box = DOMAIN, degrees: int = BLOCK_DEGREES) -> list[Box]:
    """The domain in blocks of `degrees`, from the south-west, cut at its edges."""
    found: list[Box] = []
    for south in range(int(domain.south), int(domain.north), degrees):
        for west in range(int(domain.west), int(domain.east), degrees):
            north = min(south + degrees, domain.north)
            found.append(Box(south, north, west, min(west + degrees, domain.east)))
    return found


def block_name(block: Box) -> str:
    return tile_name(int(block.south), int(block.west)).removeprefix("fine-").removesuffix(".npz")


def ghsl_tiles(box: Box) -> list[str]:
    """GHSL's 100 m tiles that a box touches, such as `R4_C19`: its points every 0.25° in
    Mollweide, whose meridians curve, so that its corners alone would not do."""
    lats = np.arange(box.south, box.north + 0.25, 0.25).clip(max=box.north)
    lons = np.arange(box.west, box.east + 0.25, 0.25).clip(max=box.east)
    lat, lon = np.meshgrid(np.radians(lats), np.radians(lons))
    theta = _mollweide_theta(lat)
    x = 2 * np.sqrt(2) / np.pi * GHSL_RADIUS_M * lon * np.cos(theta)
    y = np.sqrt(2) * GHSL_RADIUS_M * np.sin(theta)
    rows = np.floor((GHSL_NORTH_M - y) / GHSL_TILE_M).astype(np.int64) + 1
    cols = np.floor((x - GHSL_WEST_M) / GHSL_TILE_M).astype(np.int64) + 1
    found = sorted({(int(r), int(c)) for r, c in zip(rows.ravel(), cols.ravel(), strict=True)})
    return [f"R{row}_C{col}" for row, col in found]


def plan(only: str | None = None) -> list[dict[str, Any]]:
    """The blocks to build, for the workflow's matrix: each one's edges, GHSL tiles and release."""
    found = [
        {
            "name": block_name(block),
            "south": int(block.south),
            "north": int(block.north),
            "west": int(block.west),
            "east": int(block.east),
            "ghsl": " ".join(ghsl_tiles(block)),
            "release": RELEASES[block.west >= EAST_FROM],
        }
        for block in blocks()
    ]
    return [block for block in found if only in (None, block["name"])]


def build(layers_dir: Path, roads: Path, out: Path, grid: Grid) -> list[Path]:
    """The tiles from GDAL's cuts in `layers_dir` (`scripts/fine-layers.sh`: GHSL's counts per
    100 m pixel averaged onto the grid, the DEM averaged, CORINE's index at each cell) and the
    roads' GeoJSON sequence."""
    # GHSL's pixels are 100 m squares of its equal-area projection: 0.01 km² each.
    counts = np.nan_to_num(read_layer(layers_dir / "population.tif", grid)).clip(min=0)
    population = counts / GHSL_PIXEL_KM2
    surface = np.nan_to_num(read_layer(layers_dir / "built_up.tif", grid)).clip(min=0)
    built = (surface / (GHSL_PIXEL_KM2 * 1e6)).clip(0, 1)
    elevation = np.nan_to_num(read_layer(layers_dir / "elevation.tif", grid))
    corine = np.nan_to_num(read_layer(layers_dir / "corine.tif", grid))
    corine = np.where((corine >= 1) & (corine <= len(CORINE_CODES)), corine, 0)
    lengths = road_lengths(read_roads(roads), grid)
    layers: dict[str, NDArray[Any]] = {
        "corine": corine.astype(np.uint8),
        "population": population.round().clip(max=np.iinfo(np.uint16).max).astype(np.uint16),
        "built_up": (built * BUILT_UP_SCALE).round().astype(np.uint8),
        "elevation": elevation.round().astype(np.int16),
        **lengths,
    }
    for name, values in layers.items():
        logger.info("%s: max %s", name, values.max())
    return write_tiles(layers, out, grid)


def _mollweide_theta(lat: FloatArray) -> FloatArray:
    """Mollweide's auxiliary angle: 2θ + sin 2θ = π sin φ, by Newton's method."""
    theta = lat.copy()
    for _ in range(8):
        theta = theta - (2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat)) / (
            2 + 2 * np.cos(2 * theta)
        )
    return theta


def _points(coordinates: list[list[float]]) -> list[tuple[float, float]]:
    return [(float(point[0]), float(point[1])) for point in coordinates]


def _steps(line: FloatArray, grid: Grid) -> tuple[NDArray[np.intp], FloatArray]:
    """A line cut into steps of at most `STEP_M`: the cell of each step's middle (flat index)
    and its length, in metres; steps outside the grid are left out."""
    if line.shape[0] < 2:
        return np.array([], dtype=np.intp), np.array([])
    start, end = line[:-1], line[1:]
    middle_lat = np.radians((start[:, 1] + end[:, 1]) / 2)
    dx = np.radians(end[:, 0] - start[:, 0]) * np.cos(middle_lat) * EARTH_RADIUS_KM * 1000
    dy = np.radians(end[:, 1] - start[:, 1]) * EARTH_RADIUS_KM * 1000
    lengths = np.hypot(dx, dy)
    pieces = np.maximum(1, np.ceil(lengths / STEP_M)).astype(np.int64)
    segment = np.repeat(np.arange(lengths.size), pieces)
    within = np.arange(pieces.sum()) - np.repeat(np.cumsum(pieces) - pieces, pieces) + 0.5
    fraction = within / pieces[segment]
    points = start[segment] + (end[segment] - start[segment]) * fraction[:, None]
    metres = lengths[segment] / pieces[segment]
    row = np.floor((points[:, 1] - grid.box.south) * grid.cells_per_degree).astype(np.intp)
    col = np.floor((points[:, 0] - grid.box.west) * grid.cells_per_degree).astype(np.intp)
    rows, cols = grid.shape
    inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    return row[inside] * cols + col[inside], metres[inside]
