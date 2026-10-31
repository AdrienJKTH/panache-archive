import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from panache_archive import static
from panache_archive.grid import Box
from panache_archive.static import Grid

SMALL = Grid(Box(south=45.0, north=45.05, west=4.8, east=4.9), static.CELLS_PER_DEGREE)
"""6 by 12 cells near Lyon."""


def geotiff(
    path: Path,
    values: np.ndarray,
    *,
    step: float = SMALL.step,
    corner: tuple[float, float] = (SMALL.box.west, SMALL.box.north),
    nodata: str | None = None,
) -> Path:
    """A north-up GeoTIFF as GDAL writes it: the corner of pixel (0, 0) in its tiepoint."""
    tags: list[tuple[int, int, int, object, bool]] = [
        (33550, 12, 3, (step, step, 0.0), True),
        (33922, 12, 6, (0.0, 0.0, 0.0, corner[0], corner[1], 0.0), True),
    ]
    if nodata is not None:
        tags.append((42113, 2, len(nodata) + 1, nodata, True))
    tifffile.imwrite(path, values, extratags=tags)
    return path


def test_the_grid_shares_the_cams_cells_edges() -> None:
    grid = static.FRANCE_1KM

    assert grid.shape == (1260, 1860)
    # Twelve cells a CAMS cell: Lyon's CAMS cell (45.7 to 45.8° N) holds rows 564 to 575.
    assert grid.lats[564] == pytest.approx(45.7 + grid.step / 2)
    assert grid.lons[0] == pytest.approx(-5.5 + grid.step / 2)
    assert grid.lats[-1] == pytest.approx(51.5 - grid.step / 2)


def test_cells_cover_the_sphere_between_their_edges() -> None:
    grid = Grid(Box(south=0.0, north=1.0, west=0.0, east=1.0), static.CELLS_PER_DEGREE)

    areas = grid.cell_area_km2()

    # A 1° square at the equator is ~12,364 km²; its cells shrink towards the pole.
    assert areas.sum() * grid.shape[1] == pytest.approx(12_364, rel=1e-3)
    assert areas[0] > areas[-1]


def test_a_layer_is_read_from_the_south(tmp_path: Path) -> None:
    values = np.arange(72, dtype=np.float32).reshape(6, 12)

    found = static.read_layer(geotiff(tmp_path / "layer.tif", values), SMALL)

    assert found[0].tolist() == values[-1].tolist()
    assert found.dtype == np.float64


def test_no_data_becomes_nan(tmp_path: Path) -> None:
    values = np.full((6, 12), 3.0, dtype=np.float32)
    values[0, 0] = -200

    found = static.read_layer(geotiff(tmp_path / "layer.tif", values, nodata="-200"), SMALL)

    assert np.isnan(found[-1, 0])
    assert np.nansum(found) == 3.0 * 71


@pytest.mark.parametrize(
    ("shape", "step", "corner", "error"),
    [
        ((6, 11), SMALL.step, (4.8, 45.05), "not the grid's"),
        ((6, 12), 0.01, (4.8, 45.05), "pixels are"),
        ((6, 12), SMALL.step, (4.81, 45.05), "north-west corner"),
    ],
    ids=["shape", "pixels", "corner"],
)
def test_a_layer_off_the_grid_is_refused(
    tmp_path: Path, shape: tuple[int, int], step: float, corner: tuple[float, float], error: str
) -> None:
    path = geotiff(tmp_path / "layer.tif", np.zeros(shape, np.float32), step=step, corner=corner)

    with pytest.raises(ValueError, match=error):
        static.read_layer(path, SMALL)


def test_population_becomes_a_density() -> None:
    count = np.full(SMALL.shape, 100.0)
    count[0, 0] = np.nan
    count[0, 1] = -5

    density = static.population_density(count, SMALL)

    assert density[0, :2].tolist() == [0.0, 0.0]
    assert density[1, 0] == pytest.approx(100 / SMALL.cell_area_km2()[1])


def test_built_up_surface_becomes_a_share() -> None:
    area_m2 = SMALL.cell_area_km2()[0] * 1e6
    surface = np.zeros(SMALL.shape)
    surface[0, :3] = [area_m2 / 4, 2 * area_m2, np.nan]

    share = static.built_up_share(surface, SMALL)

    assert share[0, :3].tolist() == pytest.approx([0.25, 1.0, 0.0])


def test_build_writes_the_three_layers_and_what_they_are(tmp_path: Path) -> None:
    rows, cols = SMALL.shape
    population = geotiff(tmp_path / "population.tif", np.full((rows, cols), 50.0, np.float32))
    built = geotiff(tmp_path / "built_up.tif", np.zeros((rows, cols), np.float32))
    elevation = np.full((rows, cols), 170.0, np.float32)
    elevation[0, 0] = np.nan
    dem = geotiff(tmp_path / "elevation.tif", elevation, nodata="nan")

    path = static.build(population, built, dem, tmp_path / "dist", grid=SMALL)

    assert path.name == static.ASSET
    with np.load(path) as asset:
        assert sorted(asset.files) == [
            "about",
            "built_up",
            "elevation",
            "lats",
            "lons",
            "population",
        ]
        assert asset["population"].shape == SMALL.shape
        assert asset["population"].dtype == np.float32
        assert np.isnan(asset["elevation"][-1, 0])
        assert asset["lats"].tolist() == SMALL.lats.tolist()
        about = json.loads(str(asset["about"]))
    assert about["grid"]["cells_per_degree"] == 120
    assert about["units"]["population"] == "inhabitants/km²"
    assert set(about["sources"]) == {"population", "built_up", "elevation"}
