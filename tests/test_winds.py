import json
import urllib.error
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path

import numcodecs
import numpy as np
import pytest
from numpy.typing import NDArray

from panache_archive import winds

URL = "https://store.test/arco"
LATS = np.arange(90.0, -90.1, -5.0)
LONS = np.arange(0.0, 360.0, 5.0)
STORE_LEVELS = np.array([1, 500, 600, 700, 800, 850, 900, 925, 950, 975, 1000])
BOX = winds.Box(south=20.0, west=-10.0, step=5.0, rows=5, columns=6, tile=4)
BLOSC = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)


def fields(hour: int) -> dict[str, NDArray[np.float64]]:
    """Each variable over the store's grid at an hour: (level, lat, lon), or (lat, lon)."""
    level, lat, lon = np.meshgrid(STORE_LEVELS, LATS, LONS, indexing="ij")
    return {
        "u_component_of_wind": 10 + 0.01 * level + 0.1 * lat + hour % 5,
        "v_component_of_wind": -3 + 0.05 * lon + 0.0 * lat,
        "vertical_velocity": 0.004 * level / 100 - 0.001 * hour % 7,
        "surface_pressure": 101_000 - 100 * lat[0],
    }


def store(missing_from: int | None = None, values: float | None = None) -> dict[str, bytes]:
    """A tiny ARCO-ERA5 at 5°, from 2021-02-01, for one day."""
    shape = [400_000, STORE_LEVELS.size, LATS.size, LONS.size]
    meta = {
        "u_component_of_wind/.zarray": {
            "shape": shape,
            "chunks": [1, *shape[1:]],
            "compressor": BLOSC.get_config(),
        },
        "latitude/.zarray": {
            "shape": [LATS.size],
            "dtype": "<f4",
            "compressor": BLOSC.get_config(),
        },
        "longitude/.zarray": {"shape": [LONS.size], "dtype": "<f8", "compressor": None},
        "level/.zarray": {"shape": [STORE_LEVELS.size], "dtype": "<i8", "compressor": None},
    }
    files = {
        f"{URL}/.zmetadata": json.dumps({"metadata": meta}).encode(),
        f"{URL}/latitude/0": BLOSC.encode(LATS.astype("<f4").tobytes()),
        f"{URL}/longitude/0": LONS.astype("<f8").tobytes(),
        f"{URL}/level/0": STORE_LEVELS.astype("<i8").tobytes(),
    }
    first = winds.hour_of(datetime(2021, 2, 1, tzinfo=UTC))
    for hour in range(first, first + 24, 3):
        if missing_from is not None and hour >= first + missing_from:
            break
        for name, array in fields(hour).items():
            data = np.full_like(array, values) if values is not None else array
            key = f"{hour}.0.0.0" if data.ndim == 3 else f"{hour}.0.0"
            files[f"{URL}/{name}/{key}"] = BLOSC.encode(data.astype("<f4").tobytes())
    return files


def fetcher(files: dict[str, bytes]) -> winds.Fetch:
    def fetch(url: str) -> bytes:
        if url not in files:
            raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)
        return files[url]

    return fetch


def build(tmp_path: Path, files: dict[str, bytes]) -> Path:
    return winds.build(2021, 2, tmp_path, fetcher(files), box=BOX, url=URL, days=1, concurrency=2)


def test_a_day_is_built_and_read_back_tile_by_tile(tmp_path: Path) -> None:
    path = build(tmp_path, store())
    assert path.name == "winds-2021-02.winds"
    data = path.read_bytes()
    header = winds.Header.unpack(data)
    assert (header.year, header.month, header.days, header.box) == (2021, 2, 1, BOX)
    whole = np.concatenate(
        [
            np.concatenate([winds.read_tile(data, 0, row, column) for column in range(2)], axis=-1)
            for row in range(2)
        ],
        axis=-2,
    )
    assert whole.shape == (8, winds.FIELDS, 5, 6)

    first = winds.hour_of(datetime(2021, 2, 1, tzinfo=UTC))
    rows = [int(np.flatnonzero(lat == LATS)[0]) for lat in BOX.lats]
    columns = [int(np.flatnonzero(lon % 360 == LONS)[0]) for lon in BOX.lons]
    levels = [int(np.flatnonzero(level == STORE_LEVELS)[0]) for level in winds.LEVELS]
    for step in (0, 5):
        expected = fields(first + 3 * step)
        k = len(winds.LEVELS)
        for index, (name, scale) in enumerate(
            zip(winds.WINDS, (winds.WIND_SCALE, winds.WIND_SCALE, winds.OMEGA_SCALE), strict=True)
        ):
            wanted = expected[name][levels][:, rows][:, :, columns]
            got = whole[step, index * k : (index + 1) * k] * scale
            np.testing.assert_allclose(got, wanted, atol=scale / 2 + 1e-6)
        surface = whole[step, -1] * winds.PRESSURE_SCALE + winds.PRESSURE_OFFSET
        np.testing.assert_allclose(
            surface, expected["surface_pressure"][rows][:, columns], atol=winds.PRESSURE_SCALE / 2
        )


def test_a_frame_read_at_the_wrong_place_is_caught(tmp_path: Path) -> None:
    data = bytearray(build(tmp_path, store()).read_bytes())
    start = winds.OFFSET.unpack_from(data, winds.HEADER.size)[0]
    winds.FRAME_HEADER.pack_into(data, start, 0, 1, 1, 0)
    with pytest.raises(ValueError, match="is not day 0"):
        winds.read_tile(bytes(data), 0, 0, 0)


def test_the_header_is_checked() -> None:
    header = winds.Header(2021, 2, 28, winds.EUROPE)
    assert winds.Header.unpack(header.pack()) == header
    assert header.frames == 28 * 7 * 16
    with pytest.raises(ValueError, match="version 1"):
        winds.Header.unpack(b"NOTWINDS" + header.pack()[8:])


@pytest.mark.parametrize("values", [np.nan, 1e6])
def test_missing_or_huge_values_stop_the_build(tmp_path: Path, values: float) -> None:
    with pytest.raises(ValueError, match=r"missing|beyond"):
        build(tmp_path, store(values=values))


def test_a_grid_without_the_archives_points_is_refused(tmp_path: Path) -> None:
    box = winds.Box(south=21.0, west=-10.0, step=5.0, rows=2, columns=2, tile=2)
    with pytest.raises(ValueError, match="does not hold"):
        winds.build(2021, 2, tmp_path, fetcher(store()), box=box, url=URL, days=1)


def test_a_store_not_chunked_by_hour_is_refused(tmp_path: Path) -> None:
    files = store()
    meta = json.loads(files[f"{URL}/.zmetadata"])
    meta["metadata"]["u_component_of_wind/.zarray"]["chunks"][0] = 24
    files[f"{URL}/.zmetadata"] = json.dumps(meta).encode()
    with pytest.raises(ValueError, match="one chunk per hour"):
        build(tmp_path, files)


def test_encoding_round_trips_extreme_values() -> None:
    rng = np.random.default_rng(3)
    block = rng.integers(-winds.LIMIT, winds.LIMIT, size=(8, winds.FIELDS, 3, 7), dtype=np.int16)
    np.testing.assert_array_equal(winds.decoded(winds.encoded(block), block.shape), block)


def test_availability_and_plan() -> None:
    hour = winds.hour_of(datetime(2021, 2, 28, 21, tzinfo=UTC))
    files = {f"{URL}/surface_pressure/{hour}.0.0": b"x"}
    fetch = fetcher(files)
    assert winds.available(2021, 2, fetch, URL)
    assert not winds.available(2021, 3, fetch, URL)
    jobs = winds.plan([2020, 2021], {"winds-2021-01.winds"}, fetch, URL)
    assert jobs == [{"year": 2021, "months": "2"}]


def test_other_errors_are_raised() -> None:
    def fetch(url: str) -> bytes:
        raise urllib.error.HTTPError(url, 403, "Forbidden", Message(), None)

    with pytest.raises(urllib.error.HTTPError):
        winds.available(2021, 2, fetch, URL)
