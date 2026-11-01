import json
from pathlib import Path

import numpy as np
import pytest

from panache_archive import cli, fine
from panache_archive.grid import Box
from panache_archive.static import Grid
from test_static import geotiff

# Half a degree west of 5° E, a degree and a half north of 45° N: tiles W of 5° are half wide.
SMALL = Grid(Box(south=45.0, north=46.5, west=4.5, east=6.0), 20)
"""30 by 30 cells of 0.05°: tiles N45E004 (half), N45E005, N46E004 (half of both), N46E005."""


def road(layer: str, *lines: list[tuple[float, float]]) -> fine.Road:
    return fine.Road(layer, [list(line) for line in lines])


def test_a_road_is_summed_in_the_cells_it_crosses() -> None:
    # 0.05° of latitude due north at 4.525° E: ~5.56 km over one column, two rows.
    north = road("road_major", [(4.525, 45.0), (4.525, 45.1)])

    lengths = fine.road_lengths([north], SMALL)

    major = lengths["road_major"]
    assert major.dtype == np.uint16
    assert major[:, 0].tolist()[:3] == [5560, 5560, 0]
    assert major[:, 1:].sum() == 0
    assert lengths["road_primary"].sum() == 0


def test_a_long_cell_is_capped_and_steps_outside_are_left_out() -> None:
    grid = Grid(Box(south=45.0, north=45.1, west=4.5, east=4.6), 10)
    roads = [road("road_secondary", [(4.55, 45.0), (4.55, 45.1)])] * 7
    outside = road("road_secondary", [(0.0, 0.0), (0.1, 0.1)])

    lengths = fine.road_lengths([*roads, outside], grid)

    assert lengths["road_secondary"].tolist() == [[65535]]


def test_roads_are_read_from_osmiums_sequence(tmp_path: Path) -> None:
    features = [
        {"type": "Feature", "properties": {"highway": "trunk_link"},
         "geometry": {"type": "LineString", "coordinates": [[4.6, 45.1], [4.7, 45.1]]}},
        {"type": "Feature", "properties": {"highway": "primary"},
         "geometry": {"type": "MultiLineString", "coordinates": [[[4.6, 45.2], [4.6, 45.3]]]}},
        {"type": "Feature", "properties": {"highway": "residential"},
         "geometry": {"type": "LineString", "coordinates": [[4.6, 45.1], [4.7, 45.1]]}},
        {"type": "Feature", "properties": {"highway": "secondary"},
         "geometry": {"type": "Point", "coordinates": [4.6, 45.1]}},
    ]  # fmt: skip
    path = tmp_path / "roads.geojsonseq"
    path.write_text("".join(f"\x1e{json.dumps(f)}\n" for f in features) + "\n", encoding="utf-8")

    roads = list(fine.read_roads(path))

    assert [r.layer for r in roads] == ["road_major", "road_primary"]
    assert roads[1].lines == [[(4.6, 45.2), (4.6, 45.3)]]


def test_tiles_start_on_whole_degrees_and_skip_the_sea(tmp_path: Path) -> None:
    rows, cols = SMALL.shape
    corine = np.full((rows, cols), 44, dtype=np.uint8)  # the sea everywhere
    corine[:20, 10:] = 1  # land in 45-46° N, 5-6° E
    corine[25, 2] = 12  # and a field in 46-46.5° N, 4.5-5° E
    layers: dict[str, np.ndarray] = {
        "corine": corine,
        "elevation": np.arange(rows * cols).reshape(rows, cols),
    }

    written = fine.write_tiles(layers, tmp_path, SMALL)

    assert sorted(p.name for p in written) == ["fine-N45E005.npz", "fine-N46E004.npz"]
    with np.load(tmp_path / "fine-N46E004.npz") as tile:
        assert tile["corine"].shape == (10, 10)
        assert tile["elevation"][0, 0] == 20 * cols
        about = json.loads(str(tile["about"]))
    assert (about["south"], about["west"]) == (46, 4.5)
    assert about["corine_codes"][0] == 111
    with np.load(tmp_path / "fine-N45E005.npz") as tile:
        assert tile["corine"].shape == (20, 20)


def test_tile_names_say_their_south_west_corner() -> None:
    assert fine.tile_name(45, 4) == "fine-N45E004.npz"
    assert fine.tile_name(41, -6) == "fine-N41W006.npz"
    assert fine.tile_name(-3, -60) == "fine-S03W060.npz"


def cuts(folder: Path, grid: Grid) -> None:
    rows, cols = grid.shape
    corner = (grid.box.west, grid.box.north)
    folder.mkdir(parents=True, exist_ok=True)
    geotiff(
        folder / "population.tif",
        np.full((rows, cols), 0.5, np.float32),
        step=grid.step,
        corner=corner,
    )
    geotiff(
        folder / "built_up.tif",
        np.full((rows, cols), 2500.0, np.float32),
        step=grid.step,
        corner=corner,
    )
    geotiff(
        folder / "elevation.tif",
        np.full((rows, cols), 170.4, np.float32),
        step=grid.step,
        corner=corner,
    )
    corine = np.full((rows, cols), 2.0, np.float32)
    corine[-1, 0] = 48  # CORINE's no-data, in the south-west corner (rows from the north)
    geotiff(folder / "corine.tif", corine, step=grid.step, corner=corner)
    road_line = {
        "type": "Feature",
        "properties": {"highway": "motorway"},
        "geometry": {"type": "LineString", "coordinates": [[4.6, 45.1], [4.7, 45.1]]},
    }
    (folder / "roads.geojsonseq").write_text(json.dumps(road_line) + "\n", encoding="utf-8")


def test_build_turns_gdals_cuts_into_tiles(tmp_path: Path) -> None:
    cuts(tmp_path / "layers", SMALL)

    written = fine.build(
        tmp_path / "layers", tmp_path / "layers" / "roads.geojsonseq", tmp_path / "dist", SMALL
    )

    assert len(written) == 4
    with np.load(tmp_path / "dist" / "fine-N45E004.npz") as tile:
        assert tile["population"][5, 5] == 50  # 0.5 resident a 0.01 km² pixel
        assert tile["built_up"][5, 5] == 62  # 2,500 m² of 10,000: a quarter, in 250ths
        assert tile["elevation"][5, 5] == 170
        assert tile["corine"][0, 0] == 0
        assert tile["road_major"].sum() > 0
        assert set(tile.files) >= {"road_primary", "road_secondary", "about"}


def test_cli_builds_from_the_layers_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    asked: list[tuple[Path, Path, Path]] = []

    def build(layers: Path, roads: Path, out: Path) -> list[Path]:
        asked.append((layers, roads, out))
        return [out / "fine-N45E004.npz"]

    monkeypatch.setattr("panache_archive.fine.build", build)

    assert cli.main(["fine", "--layers", str(tmp_path), "--out", "dist"]) == 0
    assert asked == [(tmp_path, tmp_path / "roads.geojsonseq", Path("dist"))]
    assert "1 tiles in dist" in capsys.readouterr().out
