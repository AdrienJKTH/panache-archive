"""Read one CAMS Europe reanalysis file: the hourly field of its variable over a box.

An ADS file holds one variable over one month, `(time, lat, lon)` on the whole European grid,
such as `cams.eaq.vra.ENSa.pm10.l0.2021-02.nc` (807 MB as float32). Only the box's latitude
band is read from disk.
"""

from __future__ import annotations  # netCDF4.Variable is generic in its stubs only

import itertools
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import netCDF4
import numpy as np

from panache_archive.grid import Box, BoxCells, FloatArray, box_cells

DIMENSIONS: Final = ("time", "lat", "lon")
HOUR: Final = timedelta(hours=1)


@dataclass(frozen=True, slots=True)
class HourlyField:
    """Hourly values over a box, `values[hour, lat, lon]`, NaN where the file has no value."""

    name: str
    """The variable's name in the file, such as `pm10`."""
    units: str
    first_hour: datetime
    """Start of the first hour, timezone-aware UTC."""
    values: FloatArray
    lats: FloatArray
    """Cell centres, ascending."""
    lons: FloatArray
    """Cell centres, ascending, in [-180, 180)."""

    @property
    def hours(self) -> int:
        return int(self.values.shape[0])


def read_field(path: Path, box: Box) -> HourlyField:
    """The hourly field of the file's one data variable over `box`."""
    with netCDF4.Dataset(path) as dataset:
        name = _data_variable(dataset)
        variable = dataset.variables[name]
        cells = box_cells(
            np.asarray(dataset.variables["lat"][:], dtype=np.float64),
            np.asarray(dataset.variables["lon"][:], dtype=np.float64),
            box,
        )
        first_hour = _first_hour(dataset.variables["time"])
        values = _read_box(variable, cells)
        units = str(getattr(variable, "units", ""))
    return HourlyField(
        name=name,
        units=units,
        first_hour=first_hour,
        values=values,
        lats=cells.lats,
        lons=cells.lons,
    )


def _data_variable(dataset: netCDF4.Dataset) -> str:
    names = [n for n, v in dataset.variables.items() if tuple(v.dimensions) == DIMENSIONS]
    if len(names) != 1:
        raise ValueError(f"expected one variable over {DIMENSIONS}, found {names}")
    return names[0]


def _read_box(variable: netCDF4.Variable[Any], cells: BoxCells) -> FloatArray:
    """The box's values: its latitude band is read whole, then its longitudes are picked."""
    first, last = int(cells.lat_index.min()), int(cells.lat_index.max())
    band = np.ma.filled(np.ma.asarray(variable[:, first : last + 1, :], dtype=np.float64), np.nan)
    return np.asarray(band[:, cells.lat_index - first][:, :, cells.lon_index], dtype=np.float64)


def _first_hour(time: netCDF4.Variable[Any]) -> datetime:
    """Start of the first hour, after checking that the steps are consecutive hours."""
    stamps = netCDF4.num2date(
        time[:],
        units=time.units,
        calendar=getattr(time, "calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    listed: list[datetime] = list(stamps) if isinstance(stamps, np.ndarray) else [stamps]
    hours = [_utc(stamp) for stamp in listed]
    steps = {later - earlier for earlier, later in itertools.pairwise(hours)}
    if steps - {HOUR}:
        raise ValueError(f"time steps {sorted(steps)} are not consecutive hours")
    return hours[0]


def _utc(stamp: datetime) -> datetime:
    return stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)
