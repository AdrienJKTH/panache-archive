"""Synthetic ADS files and Open-Meteo answers for the tests."""

import json
import urllib.parse
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
from numpy.typing import NDArray

from panache_archive import ads
from panache_archive.grid import nearest, normalise_longitudes
from panache_archive.variables import ADS_NAME, Variable

FloatArray = NDArray[np.float64]

GRID_LATS = np.array([40.95, 41.05, 45.75, 51.45, 51.55])
"""A few centres of the CAMS grid: the box keeps the middle three."""
GRID_LONS = np.array([354.45, 354.55, 4.85, 9.95, 10.05])
"""In [0, 360), as some files write them: -5.55 and 10.05 lie outside the box."""
FILE_NAMES = {ADS_NAME[variable]: variable.value for variable in Variable}
"""Names of the variables inside the files, by ADS name: `pm10` for `particulate_matter_10um`."""


def synthetic(first_hour: datetime, hours: int, lats: FloatArray, lons: FloatArray) -> FloatArray:
    """Smooth hourly values that differ by hour, latitude and longitude: `(hours, lat, lon)`."""
    epoch_hours = (first_hour - datetime(2000, 1, 1, tzinfo=UTC)) // timedelta(hours=1)
    hour = (epoch_hours + np.arange(hours))[:, None, None]
    daily_cycle = 20 * np.sin(2 * np.pi * (hour % 24) / 24)
    weather = 15 * np.sin(2 * np.pi * hour / (24 * 9.3))
    place = lats[None, :, None] / 3 + lons[None, None, :] / 70
    return np.asarray(40 + daily_cycle + weather + place, dtype=np.float64)


def write_month(
    path: Path,
    name: str,
    values: FloatArray,
    first_hour: datetime,
    *,
    lats: FloatArray = GRID_LATS,
    lons: FloatArray = GRID_LONS,
    units: str = "µg/m3",
    step_hours: int = 1,
    extra_variable: bool = False,
) -> Path:
    """A netCDF file laid out as the ADS lays out its reanalysis files; NaN as fill values."""
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", values.shape[0])
        dataset.createDimension("lat", lats.size)
        dataset.createDimension("lon", lons.size)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = f"hours since {first_hour:%Y-%m-%d %H:%M:%S}"
        time[:] = np.arange(values.shape[0]) * step_hours
        dataset.createVariable("lat", "f8", ("lat",))[:] = lats
        dataset.createVariable("lon", "f8", ("lon",))[:] = lons
        variable = dataset.createVariable(name, "f4", ("time", "lat", "lon"), fill_value=-999.0)
        variable.units = units
        variable[:] = np.ma.masked_invalid(values)
        if extra_variable:
            dataset.createVariable("other", "f4", ("time", "lat", "lon"))[:] = values
    return path


@dataclass
class FakeAds:
    """Answers ADS requests with synthetic months, zipped as the ADS sends them."""

    workdir: Path
    values: Callable[[datetime, int, FloatArray, FloatArray], FloatArray] = synthetic
    zipped: bool = True
    requests: list[ads.Request] = field(default_factory=list)
    alter: Callable[[int, dict[str, Any]], None] | None = None
    """Changes the writing of a month (by its number), to make a file go wrong."""

    def __call__(self, request: ads.Request, target: Path) -> None:
        self.requests.append(request)
        year, month = int(request["year"][0]), int(request["month"][0])
        first_hour = datetime(year, month, 1, tzinfo=UTC)
        following = datetime(year + month // 12, month % 12 + 1, 1, tzinfo=UTC)
        hours = (following - first_hour) // timedelta(hours=1)
        name = FILE_NAMES[request["variable"][0]]
        options: dict[str, Any] = {"first_hour": first_hour}
        if self.alter is not None:
            self.alter(month, options)
        lats, lons = options.get("lats", GRID_LATS), options.get("lons", GRID_LONS)
        values = self.values(options["first_hour"], hours, lats, lons)
        netcdf = self.workdir / f"cams.eaq.vra.ENSa.{name}.l0.{year}-{month:02d}.nc"
        write_month(netcdf, name, values, **options)
        if not self.zipped:
            netcdf.rename(target)
            return
        with zipfile.ZipFile(target, "w") as archive:
            archive.write(netcdf, netcdf.name)
            archive.writestr("README.txt", "not a netCDF file")
        netcdf.unlink()


def open_meteo_answer(
    variable: Variable, first_hour: datetime, values: FloatArray, decimals: int = 1
) -> bytes:
    """Open-Meteo's JSON for an hourly series, values rounded as it rounds them."""
    times = [first_hour + timedelta(hours=h) for h in range(values.size)]
    return json.dumps(
        {
            "hourly": {
                "time": [f"{t:%Y-%m-%dT%H:%M}" for t in times],
                variable.value: [
                    None if np.isnan(v) else round(float(v), decimals) for v in values
                ],
            }
        }
    ).encode()


class FakeOpenMeteo:
    """Serves the synthetic hours of the nearest cell, rounded as Open-Meteo rounds them."""

    def __init__(self, variable: Variable, offset: float = 0.0, decimals: int = 1) -> None:
        self.variable, self.offset, self.decimals = variable, offset, decimals
        self.queries: list[dict[str, str]] = []

    def __call__(self, url: str) -> bytes:
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        self.queries.append(query)
        first = datetime.fromisoformat(query["start_date"]).replace(tzinfo=UTC)
        last = date.fromisoformat(query["end_date"])
        hours = ((last - first.date()).days + 1) * 24
        lat = nearest(GRID_LATS, float(query["latitude"]))
        lon = nearest(normalise_longitudes(GRID_LONS), float(query["longitude"]))
        values = synthetic(first, hours, GRID_LATS, GRID_LONS)[:, lat, lon]
        return open_meteo_answer(self.variable, first, values + self.offset, self.decimals)
