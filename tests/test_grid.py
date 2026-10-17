import numpy as np
import pytest

from panache_archive.grid import FRANCE, Box, box_cells, nearest, normalise_longitudes


def test_longitudes_read_west_of_greenwich_as_negative() -> None:
    np.testing.assert_allclose(
        normalise_longitudes(np.array([0.0, 4.85, 180.0, 354.45, 359.95, -5.45])),
        [0.0, 4.85, -180.0, -5.55, -0.05, -5.45],
        atol=1e-9,
    )


def test_the_box_keeps_the_cells_whose_centres_lie_inside() -> None:
    lats = np.array([40.95, 41.05, 45.75, 51.45, 51.55])
    lons = np.array([-5.55, -5.45, 4.85, 9.95, 10.05])

    cells = box_cells(lats, lons, FRANCE)

    np.testing.assert_array_equal(cells.lats, [41.05, 45.75, 51.45])
    np.testing.assert_array_equal(cells.lons, [-5.45, 4.85, 9.95])
    np.testing.assert_array_equal(cells.lat_index, [1, 2, 3])
    np.testing.assert_array_equal(cells.lon_index, [1, 2, 3])


def test_cells_come_in_ascending_order_whatever_the_grid_order() -> None:
    lats = np.array([51.55, 51.45, 45.75, 41.05, 40.95])  # north to south
    lons = np.array([354.45, 354.55, 359.95, 4.85, 10.05])  # 0 to 360, across Greenwich

    cells = box_cells(lats, lons, FRANCE)

    np.testing.assert_array_equal(cells.lats, [41.05, 45.75, 51.45])
    np.testing.assert_array_equal(cells.lat_index, [3, 2, 1])
    np.testing.assert_array_equal(cells.lons, [-5.45, -0.05, 4.85])
    np.testing.assert_array_equal(cells.lon_index, [1, 2, 3])


def test_centres_lose_their_float_noise() -> None:
    cells = box_cells(np.array([45.75000000001]), np.array([4.84999999]), FRANCE)

    assert cells.lats[0] == 45.75
    assert cells.lons[0] == 4.85


def test_a_box_outside_the_grid_is_an_error() -> None:
    with pytest.raises(ValueError, match="no cell"):
        box_cells(np.array([60.05]), np.array([4.85]), FRANCE)


@pytest.mark.parametrize(
    ("south", "north", "west", "east"),
    [(51.5, 41.0, -5.5, 10.0), (41.0, 91.0, -5.5, 10.0), (41.0, 51.5, 10.0, -5.5)],
)
def test_a_box_has_its_sides_in_order(south: float, north: float, west: float, east: float) -> None:
    with pytest.raises(ValueError, match="must lie"):
        Box(south=south, north=north, west=west, east=east)


def test_nearest_centre() -> None:
    centres = np.array([45.65, 45.75, 45.85])

    assert nearest(centres, 45.76) == 1
    assert nearest(np.array([1.0, 2.0, 3.0]), 1.5) == 0  # a tie goes to the first
