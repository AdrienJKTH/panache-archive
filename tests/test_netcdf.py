from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from helpers import GRID_LATS, GRID_LONS, synthetic, write_month
from panache_archive.grid import FRANCE
from panache_archive.netcdf import read_field

FEBRUARY = datetime(2021, 2, 1, tzinfo=UTC)


def test_the_field_is_read_over_the_box(tmp_path: Path) -> None:
    values = synthetic(FEBRUARY, 48, GRID_LATS, GRID_LONS)
    path = write_month(tmp_path / "pm10.nc", "pm10", values, FEBRUARY)

    field = read_field(path, FRANCE)

    assert field.name == "pm10"
    assert field.units == "µg/m3"
    assert field.first_hour == FEBRUARY
    assert field.hours == 48
    np.testing.assert_array_equal(field.lats, [41.05, 45.75, 51.45])
    np.testing.assert_array_equal(field.lons, [-5.45, 4.85, 9.95])
    # Grid longitudes 354.55, 4.85 and 9.95 are columns 1 to 3; the file is float32.
    np.testing.assert_allclose(field.values, values[:, 1:4, 1:4], rtol=1e-6)


def test_a_north_to_south_grid_is_read_south_to_north(tmp_path: Path) -> None:
    lats = GRID_LATS[::-1].copy()
    values = synthetic(FEBRUARY, 24, lats, GRID_LONS)
    path = write_month(tmp_path / "dust.nc", "dust", values, FEBRUARY, lats=lats)

    field = read_field(path, FRANCE)

    np.testing.assert_array_equal(field.lats, [41.05, 45.75, 51.45])
    np.testing.assert_allclose(field.values, values[:, 3:0:-1, 1:4], rtol=1e-6)


def test_fill_values_read_as_nan(tmp_path: Path) -> None:
    values = synthetic(FEBRUARY, 24, GRID_LATS, GRID_LONS)
    values[5, 2, 2] = np.nan
    path = write_month(tmp_path / "ozone.nc", "ozone", values, FEBRUARY)

    field = read_field(path, FRANCE)

    assert np.isnan(field.values[5, 1, 1])
    assert np.count_nonzero(np.isnan(field.values)) == 1


def test_time_steps_must_be_consecutive_hours(tmp_path: Path) -> None:
    values = synthetic(FEBRUARY, 24, GRID_LATS, GRID_LONS)
    path = write_month(tmp_path / "pm10.nc", "pm10", values, FEBRUARY, step_hours=3)

    with pytest.raises(ValueError, match="not consecutive hours"):
        read_field(path, FRANCE)


def test_a_file_holds_one_data_variable(tmp_path: Path) -> None:
    values = synthetic(FEBRUARY, 24, GRID_LATS, GRID_LONS)
    path = write_month(tmp_path / "pm10.nc", "pm10", values, FEBRUARY, extra_variable=True)

    with pytest.raises(ValueError, match="expected one variable"):
        read_field(path, FRANCE)
