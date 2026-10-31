import struct
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from panache_archive import cells
from panache_archive.grid import FloatArray

LATS = np.round(30.05 + 0.1 * np.arange(3), 2)
LONS = np.round(-24.95 + 0.1 * np.arange(19), 2)
"""Three frames a row: 8, 8 and 3 cells."""
FIRST_DAY = date(2021, 2, 1)


def _values(statistics: int = 1, days: int = 5, seed: int = 0) -> list[FloatArray]:
    rng = np.random.default_rng(seed)
    columns = []
    for _ in range(statistics):
        values = np.round(rng.gamma(2.0, 15.0, (days, LATS.size, LONS.size)), 3)
        values[1, 2, 18] = np.nan
        columns.append(values)
    return columns


def _write(path: Path, columns: list[FloatArray]) -> Path:
    cells.write(path, LATS, LONS, FIRST_DAY, columns, metadata={"variable": "pm10"})
    return path


def test_the_header_says_the_grid_and_the_days(tmp_path: Path) -> None:
    header = cells.read_header(_write(tmp_path / "a.cells", _values(statistics=2)))

    assert header.statistics == 2
    assert header.cells_per_frame == 8
    assert (header.first_lat, header.first_lon, header.step) == (30.05, -24.95, 0.1)
    assert (header.rows, header.columns, header.blocks, header.frames) == (3, 19, 3, 9)
    assert (header.first_day, header.days, header.scale) == (FIRST_DAY, 5, 100)
    np.testing.assert_array_equal(header.lats, LATS)
    np.testing.assert_array_equal(header.lons, LONS)


def test_every_cell_reads_back_as_its_hundredths(tmp_path: Path) -> None:
    columns = _values(statistics=2)
    path = _write(tmp_path / "a.cells", columns)

    header, everything = cells.read_all(path)

    expected = np.stack([cells.to_hundredths(c) for c in columns])
    np.testing.assert_array_equal(everything, expected)
    for row, column in [(0, 0), (0, 7), (1, 8), (2, 15), (2, 16), (2, 18)]:
        np.testing.assert_array_equal(
            cells.read_cell(path, row, column), expected[:, :, row, column]
        )
    assert header.frame_of(2, 18) == 8


def test_a_missing_value_is_null_then_nan(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.cells", _values())

    hundredths = cells.read_cell(path, 2, 18)

    assert hundredths[0, 1] == cells.NULL
    values = cells.to_values(hundredths)
    assert np.isnan(values[0, 1])
    assert np.isnan(values).sum() == 1


def test_hundredths_round_as_numpy_rounds() -> None:
    values = np.random.default_rng(1).uniform(0, 3000, 200_000)
    values[:1000] = np.round(values[:1000], 2) + 0.005  # halves

    hundredths = cells.to_hundredths(values)

    np.testing.assert_array_equal(
        (hundredths / 100).astype(np.float32), np.round(values, 2).astype(np.float32)
    )


def test_a_region_reads_only_its_frames(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.cells", _values(statistics=2))
    _, everything = cells.read_all(path)

    region = cells.read_region(path, range(1, 3), range(6, 17))

    np.testing.assert_array_equal(region, everything[:, :, 1:3, 6:17])


@pytest.mark.parametrize(
    ("rows", "columns"),
    [(range(0), range(3)), (range(0, 3, 2), range(3)), (range(3), range(15, 20))],
)
def test_a_region_must_be_a_block_inside_the_grid(
    tmp_path: Path, rows: range, columns: range
) -> None:
    path = _write(tmp_path / "a.cells", _values())

    with pytest.raises((ValueError, IndexError)):
        cells.read_region(path, rows, columns)


def test_a_cell_outside_the_grid_has_no_frame(tmp_path: Path) -> None:
    header = cells.read_header(_write(tmp_path / "a.cells", _values()))

    with pytest.raises(IndexError, match="outside"):
        header.frame_of(3, 0)


def test_a_frame_read_for_another_cell_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.cells", _values())
    header = cells.read_header(path)
    data = bytearray(path.read_bytes())
    # File the second frame's offsets under the first.
    first, second = header.entry_range(0)[0], header.entry_range(1)[0]
    data[first : first + 8] = data[second : second + 8]
    data[first + 8 : first + 16] = data[second + 8 : second + 16]
    path.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="was read for"):
        cells.read_cell(path, 0, 0)
    with pytest.raises(ValueError, match="is filed under"):
        cells.read_region(path, range(1), range(1))


def test_a_frame_of_other_days_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.cells", _values())
    header = cells.read_header(path)
    start, end = struct.unpack("<QQ", path.read_bytes()[64:80])
    frame = bytearray(path.read_bytes()[start:end])
    frame[12:16] = struct.pack("<I", 6)

    with pytest.raises(ValueError, match="6 days"):
        cells.decode_frame(header, bytes(frame))


def test_the_metadata_follows_the_frames(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.cells", _values())

    assert cells.read_metadata(path) == {"variable": "pm10"}


def test_frames_compress(tmp_path: Path) -> None:
    days = 365
    smooth = np.round(20 + 5 * np.sin(np.arange(days) / 9.0), 2)
    columns = [np.broadcast_to(smooth[:, None, None], (days, LATS.size, LONS.size)).copy()]

    path = _write(tmp_path / "a.cells", columns)

    assert path.stat().st_size < 0.25 * days * LATS.size * LONS.size * 4


def test_other_files_are_refused(tmp_path: Path) -> None:
    other = tmp_path / "other.cells"
    other.write_bytes(b"PAR1" + b"\0" * 80)
    old = _write(tmp_path / "old.cells", _values())
    data = bytearray(old.read_bytes())
    data[8:10] = struct.pack("<H", 9)
    old.write_bytes(bytes(data))

    with pytest.raises(ValueError, match=r"not a \.cells file"):
        cells.read_header(other)
    with pytest.raises(ValueError, match="version 9"):
        cells.read_header(old)


def test_columns_must_match_the_cells(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="do not match"):
        cells.write(tmp_path / "a.cells", LATS, LONS, FIRST_DAY, [np.zeros((2, 3, 4))], metadata={})
    with pytest.raises(ValueError, match="do not match"):
        cells.write(tmp_path / "a.cells", LATS, LONS, FIRST_DAY, [], metadata={})


def test_the_cells_must_be_a_regular_grid(tmp_path: Path) -> None:
    lats = np.array([30.05, 30.15, 30.35])

    with pytest.raises(ValueError, match="regular"):
        cells.write(
            tmp_path / "a.cells", lats, LONS, FIRST_DAY, [np.zeros((1, 3, LONS.size))], metadata={}
        )


def test_a_grid_of_two_steps_takes_the_second_version(tmp_path: Path) -> None:
    lats = np.round(30.0 + 0.5 * np.arange(3), 4)
    lons = np.round(-25.0 + 0.625 * np.arange(10), 4)
    values = np.arange(2 * 3 * 10, dtype=np.float64).reshape(2, 3, 10) / 4
    path = tmp_path / "merra2.cells"

    cells.write(path, lats, lons, FIRST_DAY, [values], metadata={}, cells_per_frame=1)

    header = cells.read_header(path)
    assert path.read_bytes()[8:10] == struct.pack("<H", 2)
    assert (header.step, header.lon_step, header.longitude_step) == (0.5, 0.625, 0.625)
    np.testing.assert_array_equal(header.lons, lons)
    np.testing.assert_array_equal(cells.read_all(path)[1][0], cells.to_hundredths(values))


def test_a_grid_of_one_step_keeps_the_first_version(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.cells", _values())

    assert path.read_bytes()[8:10] == struct.pack("<H", 1)
    assert cells.read_header(path).lon_step is None


def test_an_axis_of_one_cell_takes_the_other_s_step(tmp_path: Path) -> None:
    path = tmp_path / "row.cells"

    cells.write(path, LATS[:1], LONS, FIRST_DAY, [np.zeros((1, 1, LONS.size))], metadata={})

    assert cells.read_header(path).step == 0.1
