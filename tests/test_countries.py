import json
from pathlib import Path

import numpy as np
import pytest

from panache_archive import countries
from panache_archive.emissions import Box
from test_static import geotiff

BOX = Box(south=40.05, west=0.05, step=0.1, rows=3, columns=4, tile=10)
"""3 by 4 cells, so 30 by 40 finer ones."""
CORNER = (0.0, 40.3)


def countries_file(path: Path, rows: list[tuple[int, str, str]]) -> Path:
    features = [
        {"type": "Feature", "properties": {"NE_ID": ne_id, "ADM0_A3": iso, "NAME": name}}
        for ne_id, iso, name in rows
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def fine_raster() -> np.ndarray:
    """Rows from the north as GDAL writes them. France (NE_ID 7) fills the west, Spain (900)
    the south-east; the north-east cell is sea; one cell is split 60/40."""
    north_up = np.zeros((30, 40), dtype=np.uint32)
    north_up[:, :20] = 7
    north_up[20:, 20:] = 900
    # Cell (row 1 from the south, column 2): 60 finer cells of Spain, 40 of France.
    north_up[10:20, 20:30] = 7
    north_up[10:16, 20:30] = 900
    return north_up


def test_majority_ignores_the_sea_and_breaks_ties_low() -> None:
    fine = np.zeros((2, 4), dtype=np.uint32)
    fine[:, 0] = 3  # cell 0: half land 3, half sea: land wins
    fine[:, 2] = 2
    fine[:, 3] = 1  # cell 1: a tie between 2 and 1: 1
    np.testing.assert_array_equal(countries.majority(fine, finer=2), [[3, 1]])
    np.testing.assert_array_equal(countries.majority(np.zeros((2, 2), np.uint32), 2), [[0]])


def test_build_writes_each_cells_country(tmp_path: Path) -> None:
    raster = geotiff(tmp_path / "countries.tif", fine_raster(), step=0.01, corner=CORNER)
    geojson = countries_file(
        tmp_path / "countries.geojson",
        [(7, "FRA", "France"), (900, "ESP", "Spain"), (5, "ITA", "Italy")],
    )
    path = countries.build(raster, geojson, tmp_path / "dist", box=BOX)
    with np.load(path) as held:
        code, iso, name = held["code"], held["iso"], held["name"]
        assert (held["south"], held["west"], held["step"]) == (40.05, 0.05, 0.1)
        assert str(held["source"]).startswith("Natural Earth")
    # Only the countries present, in ADM0_A3's order, after the sea.
    assert iso.tolist() == ["", "ESP", "FRA"]
    assert name.tolist() == ["sea", "Spain", "France"]
    # Rows from the south: Spain's band, then the split row, then the north.
    np.testing.assert_array_equal(code, [[2, 2, 1, 1], [2, 2, 1, 0], [2, 2, 0, 0]])
    lines = countries.describe(path)
    assert lines[0] == "2 countries; sea 3 cells of 12."
    assert lines[1] == "FRA France: 6 cells"


def test_rasters_off_the_box_or_ids_unknown_are_refused(tmp_path: Path) -> None:
    geojson = countries_file(tmp_path / "countries.geojson", [(7, "FRA", "France")])
    for values, step, corner, message in (
        (fine_raster()[:20], 0.01, CORNER, "not 10 times"),
        (fine_raster(), 0.02, CORNER, "pixels are"),
        (fine_raster(), 0.01, (0.1, 40.3), "north-west corner"),
        (fine_raster(), 0.01, CORNER, r"NE_IDs the countries' file has not: \[900\]"),
    ):
        raster = geotiff(tmp_path / "bad.tif", values, step=step, corner=corner)
        with pytest.raises(ValueError, match=message):
            countries.build(raster, geojson, tmp_path / "dist", box=BOX)
