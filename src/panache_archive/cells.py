"""The whole domain's asset format, `.cells`: cell-major frames behind a fixed-width offset table,
so that one cell's year is read from afar in two small requests (Panache's D66).

Little-endian throughout:
- **Header**, 64 bytes (`HEADER`): magic, version, statistics, cells per frame, the grid (centre
  of the south-west cell, step, rows and columns), the first day, the days, the scale and the
  length of the metadata. A grid of two steps (MERRA-2's 0.5° by 0.625°) takes version 2
  (`HEADER_TWO_STEPS`): the latitude and longitude steps as two float32, in the place of the one
  float64 step.
- **Offset table**: one `uint64` per frame and one more, where each frame starts and where the
  last one ends. A frame holds `cells_per_frame` cells of one latitude row; the frame of row `r`
  and column `c` is `r * blocks + c // cells_per_frame`. So a cell's two offsets lie at a byte
  computed from its row and column: 16 bytes to read, then its frame.
- **Frames**: 16 bytes saying which cells and days follow (`FRAME_HEADER`), so that a reader
  checks it got the frame it asked for; then zstd over int32 hundredths of μg/m³ (`NULL` where
  missing), shaped (statistic, cell, day), byte-stream split: the first byte of every value,
  then the second, and so on, which compresses a third better than the values as they are.
- **Metadata**: JSON, after the last frame.

Hundredths are the 0.01 rounding of France's Parquet assets, so both hold the same numbers.
"""

import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, BinaryIO, Final

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from panache_archive.grid import COORDINATE_DECIMALS, FloatArray

IntArray = NDArray[np.int32]

MAGIC: Final = b"PANCELLS"
VERSION: Final = 1
HEADER: Final = struct.Struct("<8sHHHHdddIIiIII")
VERSION_TWO_STEPS: Final = 2
HEADER_TWO_STEPS: Final = struct.Struct("<8sHHHHddffIIiIII")
"""As `HEADER`, with a latitude and a longitude step (float32: exact for 0.5° and 0.625°)."""
FRAME_HEADER: Final = struct.Struct("<IIII")
"""Row, first column, cells and days of a frame."""
OFFSET: Final = struct.Struct("<Q")
CELLS_PER_FRAME: Final = 8
"""A read of ~4 KB a year, and files as small as the best Parquet layout (P3's measurements)."""
SCALE: Final = 100
"""Values are stored in hundredths."""
NULL: Final = np.iinfo(np.int32).min
ZSTD_LEVEL: Final = 3
EPOCH: Final = date(1970, 1, 1)


@dataclass(frozen=True, slots=True)
class Header:
    """What the first 64 bytes of a `.cells` file say."""

    statistics: int
    cells_per_frame: int
    first_lat: float
    first_lon: float
    step: float
    rows: int
    columns: int
    first_day: date
    days: int
    scale: int = SCALE
    metadata_length: int = 0
    lon_step: float | None = None
    """The longitude step when it differs from `step`, the latitude one (version 2)."""

    @property
    def longitude_step(self) -> float:
        return self.step if self.lon_step is None else self.lon_step

    @property
    def blocks(self) -> int:
        """Frames per latitude row."""
        return -(-self.columns // self.cells_per_frame)

    @property
    def frames(self) -> int:
        return self.rows * self.blocks

    @property
    def data_start(self) -> int:
        """Where the first frame starts."""
        return HEADER.size + OFFSET.size * (self.frames + 1)

    @property
    def lats(self) -> FloatArray:
        return _axis(self.first_lat, self.step, self.rows)

    @property
    def lons(self) -> FloatArray:
        return _axis(self.first_lon, self.longitude_step, self.columns)

    def frame_of(self, row: int, column: int) -> int:
        if not (0 <= row < self.rows and 0 <= column < self.columns):
            raise IndexError(f"cell ({row}, {column}) lies outside {self.rows} x {self.columns}")
        return row * self.blocks + column // self.cells_per_frame

    def entry_range(self, frame: int) -> tuple[int, int]:
        """The bytes holding where `frame` starts and ends: offset and length."""
        return HEADER.size + OFFSET.size * frame, 2 * OFFSET.size

    def pack(self) -> bytes:
        two_steps = self.lon_step is not None and self.lon_step != self.step
        steps = (self.step, self.longitude_step) if two_steps else (self.step,)
        return (HEADER_TWO_STEPS if two_steps else HEADER).pack(
            MAGIC,
            VERSION_TWO_STEPS if two_steps else VERSION,
            self.statistics,
            self.cells_per_frame,
            0,
            self.first_lat,
            self.first_lon,
            *steps,
            self.rows,
            self.columns,
            (self.first_day - EPOCH).days,
            self.days,
            self.scale,
            self.metadata_length,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "Header":
        if len(data) < HEADER.size or data[: len(MAGIC)] != MAGIC:
            raise ValueError("not a .cells file")
        version = struct.unpack_from("<H", data, len(MAGIC))[0]
        if version not in (VERSION, VERSION_TWO_STEPS):
            raise ValueError(f".cells version {version}; this reader knows 1 and 2")
        fields = list((HEADER if version == VERSION else HEADER_TWO_STEPS).unpack(data[:64]))
        lon_step = fields.pop(8) if version == VERSION_TWO_STEPS else None
        return cls(
            statistics=fields[2],
            cells_per_frame=fields[3],
            first_lat=fields[5],
            first_lon=fields[6],
            step=fields[7],
            rows=fields[8],
            columns=fields[9],
            first_day=EPOCH + timedelta(days=fields[10]),
            days=fields[11],
            scale=fields[12],
            metadata_length=fields[13],
            lon_step=lon_step,
        )


def write(
    path: Path,
    lats: FloatArray,
    lons: FloatArray,
    first_day: date,
    columns: Sequence[FloatArray],
    *,
    metadata: Mapping[str, Any],
    cells_per_frame: int = CELLS_PER_FRAME,
) -> None:
    """Write `columns`, `(days, lat, lon)` arrays of the statistics in order, NaN where missing;
    `cells_per_frame` cells of a row to a frame."""
    shapes = {values.shape for values in columns}
    if not columns or len(shapes) != 1 or next(iter(shapes))[1:] != (lats.size, lons.size):
        raise ValueError(f"columns {shapes} do not match {lats.size} x {lons.size} cells")
    days = next(iter(shapes))[0]
    lat_step, lon_step = _steps(lats, lons)
    # (lat, lon, statistic, day): each frame is then one contiguous slice.
    by_cell = np.ascontiguousarray(
        np.stack([to_hundredths(values) for values in columns]).transpose(2, 3, 0, 1)
    )
    header = Header(
        statistics=len(columns),
        cells_per_frame=cells_per_frame,
        first_lat=float(lats[0]),
        first_lon=float(lons[0]),
        step=lat_step,
        rows=lats.size,
        columns=lons.size,
        first_day=first_day,
        days=days,
        lon_step=lon_step if lon_step != lat_step else None,
    )
    codec = pa.Codec("zstd", compression_level=ZSTD_LEVEL)
    offsets = [header.data_start]
    with path.open("wb") as out:
        out.seek(header.data_start)
        for row in range(header.rows):
            for first in range(0, header.columns, header.cells_per_frame):
                block = by_cell[row, first : first + header.cells_per_frame]
                cells = block.shape[0]
                values = np.ascontiguousarray(block.transpose(1, 0, 2), dtype="<i4")
                payload = codec.compress(_split(values), asbytes=True)
                out.write(FRAME_HEADER.pack(row, first, cells, days) + payload)
                offsets.append(offsets[-1] + FRAME_HEADER.size + len(payload))
        text = json.dumps(metadata).encode()
        out.write(text)
        out.seek(0)
        _write_head(out, replace(header, metadata_length=len(text)), offsets)


def to_hundredths(values: FloatArray) -> IntArray:
    """Values in hundredths, as `np.round(values, 2)` rounds them; `NULL` where NaN."""
    missing = np.isnan(values)
    scaled = np.rint(np.where(missing, 0.0, values) * SCALE)
    return np.where(missing, NULL, scaled).astype(np.int32)


def to_values(hundredths: IntArray, scale: int = SCALE) -> FloatArray:
    """Hundredths back to μg/m³, NaN where `NULL`."""
    return np.where(hundredths == NULL, np.nan, hundredths / scale)


def read_header(path: Path) -> Header:
    with path.open("rb") as source:
        return Header.unpack(source.read(HEADER.size))


def read_cell(path: Path, row: int, column: int) -> IntArray:
    """One cell's hundredths, `(statistic, day)`, read as a remote reader would: its two offsets,
    then its frame."""
    header = read_header(path)
    frame = header.frame_of(row, column)
    with path.open("rb") as source:
        offset, length = header.entry_range(frame)
        source.seek(offset)
        start, end = struct.unpack("<QQ", source.read(length))
        source.seek(start)
        data = source.read(end - start)
    return cell_of_frame(header, data, row, column)


def cell_of_frame(header: Header, data: bytes, row: int, column: int) -> IntArray:
    """The cell's hundredths from its frame's bytes, `(statistic, day)`, once the frame is checked
    to be the one asked for."""
    frame_row, first, values = decode_frame(header, data)
    if (frame_row, first) != (row, column - column % header.cells_per_frame):
        raise ValueError(f"the frame of ({frame_row}, {first}) was read for ({row}, {column})")
    return np.asarray(values[:, column - first, :], dtype=np.int32)


def decode_frame(header: Header, data: bytes) -> tuple[int, int, IntArray]:
    """A frame's row, first column and hundredths, `(statistic, cell, day)`."""
    row, first, cells, days = FRAME_HEADER.unpack(data[: FRAME_HEADER.size])
    expected = min(header.cells_per_frame, header.columns - first) if first < header.columns else 0
    if days != header.days or cells != expected or cells == 0:
        raise ValueError(
            f"a frame of {cells} cells and {days} days at ({row}, {first}), "
            f"in a file of {header.days} days"
        )
    size = header.statistics * cells * days * 4
    raw = pa.Codec("zstd").decompress(data[FRAME_HEADER.size :], decompressed_size=size)
    return row, first, _join(raw).reshape(header.statistics, cells, days)


def read_all(path: Path) -> tuple[Header, IntArray]:
    """Every value of the file, `(statistic, day, lat, lon)`, as hundredths."""
    header = read_header(path)
    return header, read_region(path, range(header.rows), range(header.columns))


def read_region(path: Path, rows: range, columns: range) -> IntArray:
    """The hundredths of a block of cells, `(statistic, day, row, column)`: only its frames are
    decoded."""
    header = read_header(path)
    if not rows or not columns or rows.step != 1 or columns.step != 1:
        raise ValueError(f"rows {rows} and columns {columns} are not a block")
    header.frame_of(rows[-1], columns[-1])  # inside the grid
    data = path.read_bytes()
    per_frame = header.cells_per_frame
    out = np.empty((header.statistics, header.days, len(rows), len(columns)), dtype=np.int32)
    for row in rows:
        for block in range(columns.start // per_frame, (columns.stop - 1) // per_frame + 1):
            start, end = struct.unpack_from(
                "<QQ", data, header.entry_range(row * header.blocks + block)[0]
            )
            frame_row, first, values = decode_frame(header, data[start:end])
            if (frame_row, first) != (row, block * per_frame):
                raise ValueError(
                    f"the frame of ({frame_row}, {first}) is filed under ({row}, {block})"
                )
            low, high = max(first, columns.start), min(first + values.shape[1], columns.stop)
            out[:, :, row - rows.start, low - columns.start : high - columns.start] = values[
                :, low - first : high - first, :
            ].transpose(0, 2, 1)
    return out


def read_metadata(path: Path) -> dict[str, Any]:
    """What a `.cells` file says about itself."""
    header = read_header(path)
    with path.open("rb") as source:
        source.seek(header.entry_range(header.frames - 1)[0] + OFFSET.size)
        (end,) = OFFSET.unpack(source.read(OFFSET.size))
        source.seek(end)
        loaded: dict[str, Any] = json.loads(source.read(header.metadata_length))
    return loaded


def _write_head(out: BinaryIO, header: Header, offsets: Sequence[int]) -> None:
    out.write(header.pack())
    out.write(np.asarray(offsets, dtype="<u8").tobytes())


def _split(values: IntArray) -> bytes:
    """Byte stream split: the values' first bytes, then their second bytes, and so on."""
    return np.ascontiguousarray(values.view(np.uint8).reshape(-1, 4).T).tobytes()


def _join(data: bytes | pa.Buffer) -> IntArray:
    planes = np.frombuffer(data, dtype=np.uint8).reshape(4, -1)
    return np.ascontiguousarray(planes.T).view("<i4").reshape(-1)


def _steps(lats: FloatArray, lons: FloatArray) -> tuple[float, float]:
    """The grid's latitude and longitude steps, after checking both axes are regular and
    ascending; an axis of one cell takes the other's step."""
    lat_step, lon_step = _regular(lats), _regular(lons)
    if lat_step is None:
        if lon_step is None:
            raise ValueError("the cells are not a regular grid, ascending")
        return lon_step, lon_step
    return lat_step, lat_step if lon_step is None else lon_step


def _regular(axis: FloatArray) -> float | None:
    """The axis's one step; `None` for a single centre."""
    if axis.size < 2:
        return None
    steps = np.round(np.diff(axis), COORDINATE_DECIMALS)
    if np.unique(steps).size != 1 or steps[0] <= 0:
        raise ValueError("the cells are not a regular grid, ascending")
    return float(steps[0])


def _axis(first: float, step: float, count: int) -> FloatArray:
    return np.round(first + step * np.arange(count), COORDINATE_DECIMALS)
