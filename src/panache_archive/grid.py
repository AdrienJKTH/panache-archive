"""The archive's boxes, France and the whole domain, and where they lie in the CAMS Europe grid.

The grid is regular at 0.1°. Its centres are on x.x5° since 2020 (Lyon's cell is centred on
45.75° N, 4.85° E), and were on x.x0° until 2019 (45.8° N, 4.8° E): each file is cut on its own
grid. A cell belongs to the box when its centre lies inside.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.intp]

COORDINATE_DECIMALS: Final = 4
"""Cell centres are rounded to this many decimals, removing float noise such as 41.050000001."""


@dataclass(frozen=True, slots=True)
class Box:
    """A latitude-longitude box, in degrees north and east."""

    south: float
    north: float
    west: float
    east: float

    def __post_init__(self) -> None:
        if not -90 <= self.south < self.north <= 90:
            raise ValueError(f"south {self.south} must lie below north {self.north}")
        if not -180 <= self.west < self.east <= 180:
            raise ValueError(f"west {self.west} must lie west of east {self.east}")


FRANCE: Final = Box(south=41.0, north=51.5, west=-5.5, east=10.0)
"""Mainland France and Corsica with a margin, sea included: 105 by 155 cells."""
EUROPE: Final = Box(south=29.9, north=72.1, west=-25.1, east=45.1)
"""The whole CAMS Europe grid, 30-72° N and 25° W-45° E, edge cells of both grids included:
420 by 700 cells since 2020."""


@dataclass(frozen=True, slots=True)
class BoxCells:
    """The cells of a grid inside a box: indices into the grid's axes, and their centres.

    Both are in ascending latitude and longitude, whatever the order of the grid's axes.
    """

    lat_index: IndexArray
    lon_index: IndexArray
    lats: FloatArray
    lons: FloatArray


def normalise_longitudes(lons: FloatArray) -> FloatArray:
    """Longitudes in [-180, 180), so that 355° reads -5°."""
    return np.asarray(np.mod(lons + 180.0, 360.0) - 180.0, dtype=np.float64)


def box_cells(lats: FloatArray, lons: FloatArray, box: Box) -> BoxCells:
    """The cells of the grid with axes `lats` and `lons` whose centres lie inside `box`."""
    lats = np.round(np.asarray(lats, dtype=np.float64), COORDINATE_DECIMALS)
    lons = np.round(normalise_longitudes(np.asarray(lons, dtype=np.float64)), COORDINATE_DECIMALS)
    lat_index = _ascending(np.flatnonzero((lats > box.south) & (lats < box.north)), lats)
    lon_index = _ascending(np.flatnonzero((lons > box.west) & (lons < box.east)), lons)
    if lat_index.size == 0 or lon_index.size == 0:
        raise ValueError(f"no cell of the grid lies inside {box}")
    return BoxCells(
        lat_index=lat_index, lon_index=lon_index, lats=lats[lat_index], lons=lons[lon_index]
    )


def nearest(centres: FloatArray, value: float) -> int:
    """Index of the centre nearest to `value`; the first one on a tie."""
    return int(np.argmin(np.abs(centres - value)))


def _ascending(index: IndexArray, axis: FloatArray) -> IndexArray:
    return index[np.argsort(axis[index], kind="stable")]
