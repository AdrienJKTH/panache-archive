"""Fixed layers over France at 30 arc seconds (~1 km): what places a point within its CAMS cell,
for Panache's downscaling (Panache's P4).

The grid is 1/120° inside France's box (41 to 51.5° N, 5.5° W to 10° E): 1,260 by 1,860 cells,
twelve by twelve in each 0.1° cell of CAMS, whose edges it shares.

- **Population** (inhabitants per km²): JRC's GHS-POP, epoch 2020, release 2023A, on this very
  grid (`GHS_POP_E2020_GLOBE_R2023A_4326_30ss`): residents per cell, divided by the cell's area.
- **Built-up surface** (share of the cell): JRC's GHS-BUILT-S, epoch 2020, release 2023A
  (`GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss`): built-up m² per cell, divided by its area.
- **Elevation** (m): the Copernicus DEM GLO-90 (3 arc seconds), averaged over each cell; NaN
  where it has no tile (the sea).

GDAL cuts and averages the sources onto the grid (`scripts/static-layers.sh`); this module reads
its three GeoTIFFs, checks they lie on the grid, and writes one `.npz` asset.
"""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import tifffile
from numpy.typing import NDArray

from panache_archive.grid import FRANCE, Box

FloatArray = NDArray[np.float64]

CELLS_PER_DEGREE: Final = 120
EARTH_RADIUS_KM: Final = 6371.0088
"""The mean radius (IUGG)."""
ASSET: Final = "static-fr-1km.npz"
SOURCES: Final = {
    "population": "JRC GHS-POP R2023A, epoch 2020, 30 arc seconds, WGS 84 (CC-BY 4.0), "
    "doi:10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE",
    "built_up": "JRC GHS-BUILT-S R2023A, epoch 2020, 30 arc seconds, WGS 84 (CC-BY 4.0), "
    "doi:10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA",
    "elevation": "Copernicus DEM GLO-90 (© DLR e.V. 2010-2014 and © Airbus Defence and Space "
    "GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA)",
}
TOLERANCE_DEGREES: Final = 1e-9

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Grid:
    """A regular latitude-longitude grid over a box."""

    box: Box
    cells_per_degree: int

    @property
    def step(self) -> float:
        return 1 / self.cells_per_degree

    @property
    def shape(self) -> tuple[int, int]:
        rows = round((self.box.north - self.box.south) * self.cells_per_degree)
        cols = round((self.box.east - self.box.west) * self.cells_per_degree)
        return rows, cols

    @property
    def lats(self) -> FloatArray:
        """Centres, ascending."""
        return np.asarray(self.box.south + (np.arange(self.shape[0]) + 0.5) * self.step, np.float64)

    @property
    def lons(self) -> FloatArray:
        """Centres, ascending."""
        return np.asarray(self.box.west + (np.arange(self.shape[1]) + 0.5) * self.step, np.float64)

    def cell_area_km2(self) -> FloatArray:
        """Each row's cell area, km², from the south: the sphere's area between its edges."""
        south = np.radians(self.box.south + np.arange(self.shape[0]) * self.step)
        north = south + np.radians(self.step)
        width = np.radians(self.step)
        return np.asarray(EARTH_RADIUS_KM**2 * width * (np.sin(north) - np.sin(south)), np.float64)


FRANCE_1KM: Final = Grid(FRANCE, CELLS_PER_DEGREE)


def read_layer(path: Path, grid: Grid) -> FloatArray:
    """A single-band GeoTIFF on `grid`, rows from the south; its no-data value becomes NaN.

    It must lie on `grid`: its corner on the box's north-west corner, its pixels `grid.step`
    wide and high, and as many as the grid has.
    """
    with tifffile.TiffFile(path) as tif:
        page = tif.pages.first
        tags = page.tags
        scale = tags["ModelPixelScaleTag"].value
        tiepoint = tags["ModelTiepointTag"].value
        nodata_tag = tags.get("GDAL_NODATA")
        values = np.asarray(page.asarray(), dtype=np.float64)
    if values.shape != grid.shape:
        raise ValueError(f"{path.name} is {values.shape}, not the grid's {grid.shape}")
    if not np.allclose(scale[:2], (grid.step, grid.step), atol=TOLERANCE_DEGREES, rtol=0):
        raise ValueError(f"{path.name}'s pixels are {scale[:2]}, not {grid.step}")
    # Pixel (0, 0)'s corner, as GDAL writes rasters: x, y in the tiepoint's 4th and 5th values.
    corner = (tiepoint[3], tiepoint[4])
    if not np.allclose(corner, (grid.box.west, grid.box.north), atol=1e-6, rtol=0):
        raise ValueError(f"{path.name} starts at {corner}, not the box's north-west corner")
    if nodata_tag is not None:
        nodata = float(str(nodata_tag.value).strip("\x00 "))
        values[values == nodata] = np.nan
    return np.flipud(values)


def population_density(count: FloatArray, grid: Grid) -> FloatArray:
    """Residents per km², from residents per cell; no data counts as nobody."""
    density = np.nan_to_num(count, nan=0.0).clip(min=0) / grid.cell_area_km2()[:, None]
    return np.asarray(density, dtype=np.float64)


def built_up_share(surface_m2: FloatArray, grid: Grid) -> FloatArray:
    """The share of each cell built up, 0 to 1, from its built-up m²; no data counts as none."""
    area_m2 = grid.cell_area_km2()[:, None] * 1e6
    share = np.nan_to_num(surface_m2, nan=0.0).clip(min=0) / area_m2
    return np.asarray(share.clip(max=1.0), dtype=np.float64)


def build(
    population: Path, built_up: Path, elevation: Path, out: Path, *, grid: Grid = FRANCE_1KM
) -> Path:
    """The `.npz` asset: the three layers, rows from the south, and what they are."""
    layers = {
        "population": population_density(read_layer(population, grid), grid),
        "built_up": built_up_share(read_layer(built_up, grid), grid),
        "elevation": read_layer(elevation, grid),
    }
    about = {
        "grid": {
            "south": grid.box.south,
            "north": grid.box.north,
            "west": grid.box.west,
            "east": grid.box.east,
            "cells_per_degree": grid.cells_per_degree,
            "rows": "from the south",
        },
        "units": {"population": "inhabitants/km²", "built_up": "share", "elevation": "m"},
        "sources": SOURCES,
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / ASSET
    arrays: dict[str, Any] = {name: values.astype(np.float32) for name, values in layers.items()}
    np.savez_compressed(
        path, lats=grid.lats, lons=grid.lons, about=np.array(json.dumps(about)), **arrays
    )
    for name, values in layers.items():
        missing = int(np.isnan(values).sum())
        logger.info("%s: mean %.3g, %d cells without a value", name, np.nanmean(values), missing)
    return path
