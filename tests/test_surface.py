import json
import urllib.error
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path

import numcodecs
import numpy as np
import pytest
from numpy.typing import NDArray

from panache_archive import cli, surface, winds

URL = "https://store.test/arco"
LATS = np.arange(90.0, -90.1, -5.0)
LONS = np.arange(0.0, 360.0, 5.0)
BOX = winds.Box(south=20.0, west=-10.0, step=5.0, rows=5, columns=6, tile=4)
BLOSC = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
FIRST = winds.hour_of(datetime(2021, 2, 1, tzinfo=UTC))


def fields(hour: int) -> dict[str, NDArray[np.float64]]:
    """Each variable over the store's grid at an hour: (lat, lon)."""
    lat, lon = np.meshgrid(LATS, LONS, indexing="ij")
    return {
        "boundary_layer_height": 800 + 5 * lat + 30.0 * (hour % 24),
        "friction_velocity": 0.3 + 0.001 * lon + 0.0 * lat,
        "instantaneous_surface_sensible_heat_flux": -100 + lat + 4.0 * (hour % 24),
        "2m_temperature": 280 + 0.1 * lat + 0.0 * lon,
    }


def store(hours: int = 24, values: float | None = None) -> dict[str, bytes]:
    """A tiny ARCO-ERA5 at 5°, from 2021-02-01."""
    shape = [400_000, LATS.size, LONS.size]
    array = {"shape": shape, "chunks": [1, *shape[1:]], "compressor": BLOSC.get_config()}
    meta = {
        "boundary_layer_height/.zarray": array,
        "latitude/.zarray": {"shape": [LATS.size], "dtype": "<f4", "compressor": None},
        "longitude/.zarray": {"shape": [LONS.size], "dtype": "<f8", "compressor": None},
    }
    files = {
        f"{URL}/.zmetadata": json.dumps({"metadata": meta}).encode(),
        f"{URL}/latitude/0": LATS.astype("<f4").tobytes(),
        f"{URL}/longitude/0": LONS.astype("<f8").tobytes(),
    }
    for hour in range(FIRST, FIRST + hours):
        for name, data in fields(hour).items():
            kept = np.full_like(data, values) if values is not None else data
            files[f"{URL}/{name}/{hour}.0.0"] = BLOSC.encode(kept.astype("<f4").tobytes())
    return files


def fetcher(files: dict[str, bytes]) -> winds.Fetch:
    def fetch(url: str) -> bytes:
        if url not in files:
            raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)
        return files[url]

    return fetch


def build(tmp_path: Path, files: dict[str, bytes]) -> Path:
    return surface.build(2021, 2, tmp_path, fetcher(files), box=BOX, url=URL, days=1, concurrency=2)


def test_a_day_is_built_and_read_back_tile_by_tile(tmp_path: Path) -> None:
    path = build(tmp_path, store())
    assert path.name == "surface-2021-02.surface"
    data = path.read_bytes()
    header = surface.Header.unpack(data)
    assert (header.year, header.month, header.days, header.box) == (2021, 2, 1, BOX)
    assert [field.name for field in header.fields] == [field.name for field in surface.FIELDS]
    whole = np.concatenate(
        [
            np.concatenate(
                [surface.read_tile(data, 0, row, column) for column in range(2)], axis=-1
            )
            for row in range(2)
        ],
        axis=-2,
    )
    assert whole.shape == (24, len(surface.FIELDS), 5, 6)
    rows = [int(np.flatnonzero(lat == LATS)[0]) for lat in BOX.lats]
    columns = [int(np.flatnonzero(lon % 360 == LONS)[0]) for lon in BOX.lons]
    for step in (0, 13, 23):
        expected = fields(FIRST + step)
        for index, field in enumerate(header.fields):
            got = whole[step, index] * field.scale + field.offset
            wanted = expected[field.name][rows][:, columns]
            np.testing.assert_allclose(got, wanted, atol=field.scale / 2 + 1e-4)


def test_the_header_keeps_each_fields_scale() -> None:
    header = surface.Header(2021, 2, 28, winds.EUROPE)
    again = surface.Header.unpack(header.pack())
    assert (again.year, again.month, again.days, again.box) == (2021, 2, 28, winds.EUROPE)
    for field, kept in zip(again.fields, surface.FIELDS, strict=True):
        assert field.name == kept.name
        assert field.scale == pytest.approx(kept.scale)
        assert field.offset == pytest.approx(kept.offset)
    assert header.frames == 28 * 7 * 16
    with pytest.raises(ValueError, match="version 1"):
        surface.Header.unpack(b"NOTSURFC" + header.pack()[8:])


def test_a_frame_read_at_the_wrong_place_is_caught(tmp_path: Path) -> None:
    data = bytearray(build(tmp_path, store()).read_bytes())
    start = winds.OFFSET.unpack_from(data, surface.HEADER.size)[0]
    winds.FRAME_HEADER.pack_into(data, start, 0, 1, 1, 0)
    with pytest.raises(ValueError, match="is not day 0"):
        surface.read_tile(bytes(data), 0, 0, 0)


@pytest.mark.parametrize("values", [np.nan, 1e6])
def test_missing_or_huge_values_stop_the_build(tmp_path: Path, values: float) -> None:
    with pytest.raises(ValueError, match=r"missing|beyond"):
        build(tmp_path, store(values=values))


def test_a_store_not_chunked_by_hour_is_refused(tmp_path: Path) -> None:
    files = store()
    meta = json.loads(files[f"{URL}/.zmetadata"])
    meta["metadata"]["boundary_layer_height/.zarray"]["chunks"][0] = 24
    files[f"{URL}/.zmetadata"] = json.dumps(meta).encode()
    with pytest.raises(ValueError, match="one chunk per hour"):
        build(tmp_path, files)


def test_candidates_are_measured() -> None:
    candidates = (
        surface.Field("boundary_layer_height", 1.0),
        surface.Field("boundary_layer_height", 4.0),
        surface.Field("2m_temperature", 0.01, 273.15),
    )
    rows = surface.measure(
        2021, 2, 1, fetcher(store()), box=BOX, url=URL, candidates=candidates, concurrency=2
    )
    assert [(row["field"], row["scale"]) for row in rows] == [
        ("boundary_layer_height", 1.0),
        ("boundary_layer_height", 4.0),
        ("2m_temperature", 0.01),
    ]
    assert rows[0]["min"] == pytest.approx(800 + 5 * 20)
    assert rows[0]["max"] == pytest.approx(800 + 5 * 40 + 30 * 23)
    assert rows[2]["min"] == pytest.approx(282, abs=0.01)
    assert rows[0]["gb_per_year"] == pytest.approx(rows[0]["bytes"] * 365.25 / 1e9)
    assert rows[1]["bytes"] <= rows[0]["bytes"]


def test_availability_and_plan() -> None:
    hour = winds.hour_of(datetime(2021, 2, 28, 23, tzinfo=UTC))
    fetch = fetcher({f"{URL}/boundary_layer_height/{hour}.0.0": b"x"})
    assert surface.available(2021, 2, fetch, URL)
    assert not surface.available(2021, 3, fetch, URL)
    jobs = surface.plan([2020, 2021], {"surface-2021-01.surface"}, fetch, URL)
    assert jobs == [{"year": 2021, "months": "2"}]


def test_other_errors_are_raised() -> None:
    def fetch(url: str) -> bytes:
        raise urllib.error.HTTPError(url, 403, "Forbidden", Message(), None)

    with pytest.raises(urllib.error.HTTPError):
        surface.available(2021, 2, fetch, URL)


def test_the_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fetch = fetcher(store())
    out = tmp_path / "dist"

    def fake_plan(years: list[int], built: set[str]) -> list[dict[str, object]]:
        return [{"years": years, "built": len(built)}]

    def fake_build(year: int, month: int, out_dir: Path, *, days: int | None = None) -> Path:
        return real_build(year, month, out_dir, fetch, box=BOX, url=URL, days=days)

    def fake_measure(year: int, month: int, days: int) -> list[dict[str, object]]:
        return real_measure(year, month, days, fetch, box=BOX, url=URL, candidates=surface.FIELDS)

    real_build, real_measure = surface.build, surface.measure

    monkeypatch.setattr(surface, "plan", fake_plan)
    monkeypatch.setattr(surface, "build", fake_build)
    monkeypatch.setattr(surface, "measure", fake_measure)

    built = tmp_path / "built.txt"
    built.write_text("surface-2021-01.surface", encoding="utf-8")
    assert cli.main(["surface-plan", "--years", "2021", "--built", str(built)]) == 0
    assert json.loads(capsys.readouterr().out) == [{"years": [2021], "built": 1}]

    arguments = ["--year", "2021", "--month", "2", "--days", "1", "--out", str(out)]
    assert cli.main(["surface-build", *arguments]) == 0
    assert capsys.readouterr().out.strip().endswith("surface-2021-02.surface")
    assert (out / "surface-2021-02.surface").stat().st_size > surface.HEADER.size

    assert cli.main(["surface-pilot", *arguments]) == 0
    table = capsys.readouterr().out.splitlines()
    assert table[0].startswith("| Field |")
    assert len(table) == 2 + len(surface.FIELDS)
