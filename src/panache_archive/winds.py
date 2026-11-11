"""ERA5's winds on pressure levels for Panache's back-trajectories (Panache's D83).

A trajectory follows u, v and the pressure velocity omega for 96 hours. ARCO-ERA5 (Google's Zarr
copy of ERA5, public and anonymous) stores one hour of the whole globe on 37 levels per chunk, so
a trajectory read from it would cost ~15 GB: this archive keeps what trajectories over Europe need,
readable a few tiles at a time.

- **Content**: u and v (m/s) and omega (Pa/s) on 10 levels from 500 to 1000 hPa, and the surface
  pressure (Pa), every 3 hours, at ERA5's 0.25° points one in two each way (0.5°, not averaged),
  over 15-84.5° N and 70° W-89.5° E: 140 rows by 320 columns.
- **Values**: int16, u and v in tenths of m/s, omega in units of 0.002 Pa/s, the surface pressure
  in units of 2 Pa around 80,000 Pa. Nothing reached the limits in Panache's pilot.
- **Assets**: one per month, `winds-2021-02.winds`.

The format, little-endian:
- **Header**, 64 bytes (`HEADER`): magic, version, year, month, days, steps a day, fields,
  levels, rows, columns, tile side, then as float32 the south and west centres, the step, and the
  scales of the winds, omega and the surface pressure, and the latter's offset.
- **Offset table**: one `uint64` per frame and one more, where each frame starts and where the
  last one ends. The frame of day `d` and tile `(r, c)` is `(d * tile_rows + r) * tile_columns
  + c`: a tile's two offsets lie at a byte computed from its day and place.
- **Frames**: 8 bytes saying which day and tile follow (`FRAME_HEADER`), then zstd over the int16
  values shaped (step, field, row, column), each row differenced along longitude and byte split
  (low bytes, then high bytes): 20 % smaller than the values as they are.
  Fields are u on the 10 levels, then v, then omega, then the surface pressure.

Data: Hersbach et al. (2020), ERA5, doi:10.1002/qj.3803; ARCO-ERA5, Carver and Merose (2023),
https://github.com/google-research/arco-era5.
"""

import calendar
import concurrent.futures
import json
import logging
import struct
import urllib.error
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numcodecs
import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from panache_archive import web

type Int16Array = NDArray[np.int16]
type FloatArray = NDArray[np.float64]
Fetch = Callable[[str], bytes]
"""Fetches a URL and returns the body; raises `urllib.error.HTTPError`."""

logger = logging.getLogger(__name__)

STORE: Final = (
    "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)
WINDS: Final = ("u_component_of_wind", "v_component_of_wind", "vertical_velocity")
SURFACE: Final = "surface_pressure"
EPOCH: Final = datetime(1900, 1, 1, tzinfo=UTC)
"""ARCO-ERA5's first hour: its chunk `n` is hour `n` after it."""
LEVELS: Final = (500, 600, 700, 800, 850, 900, 925, 950, 975, 1000)
"""hPa, from the top down: the order of Panache's `WindField`."""
FIELDS: Final = 3 * len(LEVELS) + 1
STEP_HOURS: Final = 3
STEPS_PER_DAY: Final = 24 // STEP_HOURS
WIND_SCALE: Final = 0.1
OMEGA_SCALE: Final = 0.002
PRESSURE_SCALE: Final = 2.0
PRESSURE_OFFSET: Final = 80_000.0
LIMIT: Final = 32_767
MAGIC: Final = b"PANWINDS"
VERSION: Final = 1
HEADER: Final = struct.Struct("<8s10H7f8x")
FRAME_HEADER: Final = struct.Struct("<4H")
"""Day, tile row, tile column, and zero."""
OFFSET: Final = struct.Struct("<Q")
ZSTD_LEVEL: Final = 3
CONCURRENCY: Final = 8
FIRST_YEAR: Final = 2013


@dataclass(frozen=True, slots=True)
class Box:
    """The archive's grid: centres from (south, west) every `step` degrees, cut in square tiles."""

    south: float
    west: float
    step: float
    rows: int
    columns: int
    tile: int

    @property
    def lats(self) -> FloatArray:
        return np.asarray(self.south + self.step * np.arange(self.rows), dtype=np.float64)

    @property
    def lons(self) -> FloatArray:
        return np.asarray(self.west + self.step * np.arange(self.columns), dtype=np.float64)

    @property
    def tile_rows(self) -> int:
        return -(-self.rows // self.tile)

    @property
    def tile_columns(self) -> int:
        return -(-self.columns // self.tile)


EUROPE: Final = Box(south=15.0, west=-70.0, step=0.5, rows=140, columns=320, tile=20)


@dataclass(frozen=True, slots=True)
class Header:
    year: int
    month: int
    days: int
    box: Box

    @property
    def frames(self) -> int:
        return self.days * self.box.tile_rows * self.box.tile_columns

    def frame(self, day: int, tile_row: int, tile_column: int) -> int:
        return (day * self.box.tile_rows + tile_row) * self.box.tile_columns + tile_column

    def pack(self) -> bytes:
        box = self.box
        return HEADER.pack(
            MAGIC,
            VERSION,
            self.year,
            self.month,
            self.days,
            STEPS_PER_DAY,
            FIELDS,
            len(LEVELS),
            box.rows,
            box.columns,
            box.tile,
            box.south,
            box.west,
            box.step,
            WIND_SCALE,
            OMEGA_SCALE,
            PRESSURE_SCALE,
            PRESSURE_OFFSET,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "Header":
        magic, version, year, month, days, steps, fields, levels, rows, columns, tile, *rest = (
            HEADER.unpack(data[: HEADER.size])
        )
        if (magic, version, steps, fields, levels) != (
            MAGIC,
            VERSION,
            STEPS_PER_DAY,
            FIELDS,
            len(LEVELS),
        ):
            raise ValueError("not a version 1 `.winds` file of this layout")
        south, west, step = rest[:3]
        return cls(year, month, days, Box(south, west, step, rows, columns, tile))


@dataclass(frozen=True, slots=True)
class Store:
    """Where ARCO-ERA5 is, and what its metadata says: the codec, and the rows, columns and
    levels this archive takes from its chunks."""

    url: str
    codec: Any
    grid: tuple[int, int]
    rows: NDArray[np.intp]
    columns: NDArray[np.intp]
    levels: NDArray[np.intp]

    @classmethod
    def opened(cls, fetch: Fetch, box: Box = EUROPE, url: str = STORE) -> "Store":
        meta = json.loads(fetch(f"{url}/.zmetadata"))["metadata"]
        array = meta[f"{WINDS[0]}/.zarray"]
        if array["chunks"][0] != 1 or array["chunks"][1:] != array["shape"][1:]:
            raise ValueError(f"expected one chunk per hour, got {array['chunks']}")
        lats = _coordinate(fetch, url, meta, "latitude")
        lons = _coordinate(fetch, url, meta, "longitude")
        levels = _coordinate(fetch, url, meta, "level")
        return cls(
            url=url,
            codec=numcodecs.get_codec(array["compressor"]),
            grid=(array["shape"][-2], array["shape"][-1]),
            rows=_positions(lats, box.lats),
            columns=_positions(lons, box.lons % 360),
            levels=_positions(levels, np.array(LEVELS, dtype=np.float64)),
        )

    def step(self, hour: int, fetch: Fetch) -> Int16Array:
        """One hour as (field, row, column) int16."""
        fields = []
        for variable, scale in zip(WINDS, (WIND_SCALE, WIND_SCALE, OMEGA_SCALE), strict=True):
            values = self._chunk(fetch, f"{variable}/{hour}.0.0.0")[self.levels]
            fields.append(_quantised(values[:, self.rows][:, :, self.columns], scale, 0.0))
        surface = self._chunk(fetch, f"{SURFACE}/{hour}.0.0")
        fields.append(
            _quantised(surface[:, self.rows][:, :, self.columns], PRESSURE_SCALE, PRESSURE_OFFSET)
        )
        return np.concatenate(fields)

    def _chunk(self, fetch: Fetch, key: str) -> NDArray[np.float32]:
        raw = self.codec.decode(fetch(f"{self.url}/{key}"))
        return np.frombuffer(raw, dtype="<f4").reshape(-1, *self.grid)


def _coordinate(fetch: Fetch, url: str, meta: dict[str, Any], name: str) -> FloatArray:
    array = meta[f"{name}/.zarray"]
    raw = fetch(f"{url}/{name}/0")
    codec = numcodecs.get_codec(array["compressor"]) if array["compressor"] else None
    values = np.frombuffer(codec.decode(raw) if codec else raw, dtype=array["dtype"])
    return values[: array["shape"][0]].astype(np.float64)


def _positions(axis: FloatArray, wanted: FloatArray) -> NDArray[np.intp]:
    """Where each wanted coordinate lies on the axis, which must hold it exactly."""
    found = np.array(
        [np.flatnonzero(np.abs(axis - value) < 1e-6) for value in wanted], dtype=object
    )
    if any(len(positions) != 1 for positions in found):
        raise ValueError("the store's grid does not hold the archive's points")
    return np.array([int(positions[0]) for positions in found], dtype=np.intp)


def _quantised(values: NDArray[np.float32], scale: float, offset: float) -> Int16Array:
    if not np.isfinite(values).all():
        raise ValueError("missing values in ARCO-ERA5")
    scaled = np.round((values.astype(np.float64) - offset) / scale)
    if np.abs(scaled).max() > LIMIT:
        raise ValueError(f"values beyond int16 at a scale of {scale}")
    return scaled.astype(np.int16)


def hour_of(moment: datetime) -> int:
    return int((moment - EPOCH).total_seconds()) // 3600


def encoded(block: Int16Array) -> bytes:
    """A tile's (step, field, row, column) values, differenced along longitude, byte split and
    compressed."""
    deltas = np.diff(block, axis=-1, prepend=np.zeros_like(block[..., :1]))
    split = np.ascontiguousarray(deltas).view(np.uint8).reshape(-1, 2).T.tobytes()
    return bytes(pa.Codec("zstd", compression_level=ZSTD_LEVEL).compress(split, asbytes=True))


def decoded(data: bytes, shape: tuple[int, int, int, int]) -> Int16Array:
    raw = pa.Codec("zstd").decompress(data, decompressed_size=2 * int(np.prod(shape)), asbytes=True)
    deltas = np.frombuffer(raw, dtype=np.uint8).reshape(2, -1).T.copy().view("<i2")
    return np.cumsum(deltas.reshape(shape), axis=-1, dtype=np.int16)


def write(path: Path, header: Header, month: Int16Array) -> None:
    """Writes a month shaped (step, field, row, column)."""
    box = header.box
    frames = []
    for day in range(header.days):
        steps = month[day * STEPS_PER_DAY : (day + 1) * STEPS_PER_DAY]
        for tile_row in range(box.tile_rows):
            for tile_column in range(box.tile_columns):
                block = steps[
                    :,
                    :,
                    tile_row * box.tile : (tile_row + 1) * box.tile,
                    tile_column * box.tile : (tile_column + 1) * box.tile,
                ]
                frames.append(FRAME_HEADER.pack(day, tile_row, tile_column, 0) + encoded(block))
    start = HEADER.size + OFFSET.size * (len(frames) + 1)
    offsets = np.concatenate([[0], np.cumsum([len(frame) for frame in frames])]) + start
    with path.open("wb") as file:
        file.write(header.pack())
        file.write(offsets.astype("<u8").tobytes())
        for frame in frames:
            file.write(frame)


def read_tile(data: bytes, day: int, tile_row: int, tile_column: int) -> Int16Array:
    """A tile's day from a whole `.winds` file: (step, field, row, column)."""
    header = Header.unpack(data)
    box = header.box
    index = header.frame(day, tile_row, tile_column)
    start, end = (
        OFFSET.unpack_from(data, HEADER.size + OFFSET.size * i)[0] for i in (index, index + 1)
    )
    if FRAME_HEADER.unpack_from(data, start) != (day, tile_row, tile_column, 0):
        raise ValueError(f"frame {index} is not day {day}, tile ({tile_row}, {tile_column})")
    rows = min(box.tile, box.rows - tile_row * box.tile)
    columns = min(box.tile, box.columns - tile_column * box.tile)
    return decoded(data[start + FRAME_HEADER.size : end], (STEPS_PER_DAY, FIELDS, rows, columns))


def build(
    year: int,
    month: int,
    out_dir: Path,
    fetch: Fetch = web.read,
    *,
    box: Box = EUROPE,
    url: str = STORE,
    days: int | None = None,
    concurrency: int = CONCURRENCY,
) -> Path:
    """Builds `winds-YYYY-MM.winds` from ARCO-ERA5; `days` cuts the month short (pilots, tests)."""
    store = Store.opened(fetch, box, url)
    count = days or calendar.monthrange(year, month)[1]
    first = hour_of(datetime(year, month, 1, tzinfo=UTC))
    hours = [first + STEP_HOURS * step for step in range(count * STEPS_PER_DAY)]
    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        values = np.stack(list(pool.map(lambda hour: store.step(hour, fetch), hours)))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"winds-{year}-{month:02d}.winds"
    write(path, Header(year, month, count, box), values)
    logger.info("%s: %d steps, %.0f MB", path.name, len(hours), path.stat().st_size / 1e6)
    return path


def available(year: int, month: int, fetch: Fetch = web.read, url: str = STORE) -> bool:
    """Whether ARCO-ERA5 holds the month's last step yet."""
    last = datetime(year, month, calendar.monthrange(year, month)[1], 21, tzinfo=UTC)
    try:
        fetch(f"{url}/{SURFACE}/{hour_of(last)}.0.0")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise
    return True


def plan(
    years: Sequence[int], built: set[str], fetch: Fetch = web.read, url: str = STORE
) -> list[dict[str, Any]]:
    """The matrix: each year with months still missing and out, and those months."""
    jobs = []
    for year in years:
        months = [
            month
            for month in range(1, 13)
            if f"winds-{year}-{month:02d}.winds" not in built and available(year, month, fetch, url)
        ]
        if months:
            jobs.append({"year": year, "months": " ".join(str(month) for month in months)})
    return jobs
