"""ERA5's boundary layer for Panache's Lagrangian footprints (Panache's D86).

A footprint dilutes surface emissions in the boundary layer: it needs the layer's height and what
mixes it, hour by hour, wherever a cloud of particles passes. The winds archive (`winds`) holds
neither. ARCO-ERA5 stores one hour of the whole globe per chunk for each of these fields; this
archive keeps Europe's, readable a few tiles at a time, beside the `.winds` files and on their
grid and tiles.

- **Content** (`FIELDS`), every hour, at ERA5's 0.25° points one in two each way (0.5°, not
  averaged), over 15-84.5° N and 70° W-89.5° E, 140 rows by 320 columns:
  - the boundary layer's height, m;
  - the friction velocity, m/s;
  - the instantaneous surface sensible heat flux, W/m², ECMWF's sign: **positive downward**, so
    negative when the ground heats the air.
- **Values**: int16, each field with its own scale and offset, written in the header: the
  height to the metre, the friction velocity to 0.005 m/s, the heat flux to 2 W/m².
- **Assets**: one per month, `surface-2021-02.surface` (~85 MB), about 1 GB a year.

The format is the `.winds` files', little-endian:
- **Header**, 128 bytes (`HEADER`): magic, version, year, month, days, steps a day, fields, rows,
  columns, tile side, then as float32 the south and west centres and the step, then each field's
  scale and offset (up to `MAX_FIELDS`).
- **Offset table**: one `uint64` per frame and one more. The frame of day `d` and tile `(r, c)`
  is `(d * tile_rows + r) * tile_columns + c`.
- **Frames**: 8 bytes saying which day and tile follow, then zstd over the int16 values shaped
  (step, field, row, column), differenced along longitude and byte split.

Data: Hersbach et al. (2020), ERA5, doi:10.1002/qj.3803; ARCO-ERA5, Carver and Merose (2023),
https://github.com/google-research/arco-era5.
"""

import calendar
import concurrent.futures
import json
import logging
import struct
import urllib.error
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numcodecs
import numpy as np

from panache_archive import web
from panache_archive.winds import (
    EUROPE,
    FRAME_HEADER,
    OFFSET,
    STORE,
    Box,
    Fetch,
    Int16Array,
    _coordinate,
    _positions,
    _quantised,
    decoded,
    encoded,
    hour_of,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Field:
    """One of ARCO-ERA5's single-level variables, and how it is kept: value = int16 * scale +
    offset."""

    name: str
    scale: float
    offset: float = 0.0


FIELDS: Final = (
    Field("boundary_layer_height", 1.0),
    Field("friction_velocity", 0.005),
    Field("instantaneous_surface_sensible_heat_flux", 2.0),
)
"""Measured by the pilot over 1-4 February 2021 (panache-archive#54): 0.44, 0.30 and 0.26 GB a
year. The height sets a footprint's dilution, so it keeps its metre; the two others only set how
fast the layer mixes, which a footprint barely feels (4 % for a factor of four)."""
CANDIDATES: Final = (
    Field("boundary_layer_height", 1.0),
    Field("boundary_layer_height", 4.0),
    Field("friction_velocity", 0.001),
    Field("friction_velocity", 0.002),
    Field("friction_velocity", 0.005),
    Field("instantaneous_surface_sensible_heat_flux", 0.1),
    Field("instantaneous_surface_sensible_heat_flux", 0.5),
    Field("instantaneous_surface_sensible_heat_flux", 2.0),
    Field("2m_temperature", 0.01, 273.15),
    Field("2m_temperature", 0.1, 273.15),
    Field("10m_u_component_of_wind", 0.1),
    Field("10m_v_component_of_wind", 0.1),
    Field("100m_u_component_of_wind", 0.1),
    Field("100m_v_component_of_wind", 0.1),
)
"""What the pilot measures before the archive's content is fixed."""
STEPS_PER_DAY: Final = 24
MAX_FIELDS: Final = 8
MAGIC: Final = b"PANSURFC"
VERSION: Final = 1
HEADER: Final = struct.Struct(f"<8s9H3f{2 * MAX_FIELDS}f26x")
CONCURRENCY: Final = 8
FIRST_YEAR: Final = 2013


@dataclass(frozen=True, slots=True)
class Header:
    year: int
    month: int
    days: int
    box: Box
    fields: tuple[Field, ...] = FIELDS

    @property
    def frames(self) -> int:
        return self.days * self.box.tile_rows * self.box.tile_columns

    def frame(self, day: int, tile_row: int, tile_column: int) -> int:
        return (day * self.box.tile_rows + tile_row) * self.box.tile_columns + tile_column

    def pack(self) -> bytes:
        box = self.box
        scales = [0.0] * (2 * MAX_FIELDS)
        for index, field in enumerate(self.fields):
            scales[2 * index : 2 * index + 2] = [field.scale, field.offset]
        return HEADER.pack(
            MAGIC,
            VERSION,
            self.year,
            self.month,
            self.days,
            STEPS_PER_DAY,
            len(self.fields),
            box.rows,
            box.columns,
            box.tile,
            box.south,
            box.west,
            box.step,
            *scales,
        )

    @classmethod
    def unpack(cls, data: bytes, names: Sequence[str] = tuple(f.name for f in FIELDS)) -> "Header":
        """The header; the fields' names are the version's, their scales the file's."""
        magic, version, year, month, days, steps, count, rows, columns, tile, *rest = HEADER.unpack(
            data[: HEADER.size]
        )
        if (magic, version, steps, count) != (MAGIC, VERSION, STEPS_PER_DAY, len(names)):
            raise ValueError("not a version 1 `.surface` file of this layout")
        south, west, step = rest[:3]
        fields = tuple(
            Field(name, rest[3 + 2 * index], rest[4 + 2 * index])
            for index, name in enumerate(names)
        )
        return cls(year, month, days, Box(south, west, step, rows, columns, tile), fields)


@dataclass(frozen=True, slots=True)
class Store:
    """Where ARCO-ERA5 is, its codec, and the rows and columns this archive takes."""

    url: str
    codec: Any
    grid: tuple[int, int]
    rows: Any
    columns: Any

    @classmethod
    def opened(cls, fetch: Fetch, box: Box = EUROPE, url: str = STORE) -> "Store":
        meta = json.loads(fetch(f"{url}/.zmetadata"))["metadata"]
        array = meta[f"{FIELDS[0].name}/.zarray"]
        if array["chunks"][0] != 1 or array["chunks"][1:] != array["shape"][1:]:
            raise ValueError(f"expected one chunk per hour, got {array['chunks']}")
        lats = _coordinate(fetch, url, meta, "latitude")
        lons = _coordinate(fetch, url, meta, "longitude")
        return cls(
            url=url,
            codec=numcodecs.get_codec(array["compressor"]),
            grid=(array["shape"][-2], array["shape"][-1]),
            rows=_positions(lats, box.lats),
            columns=_positions(lons, box.lons % 360),
        )

    def hour(self, hour: int, fields: Sequence[Field], fetch: Fetch) -> Int16Array:
        """One hour as (field, row, column) int16."""
        kept = []
        for field in fields:
            raw = self.codec.decode(fetch(f"{self.url}/{field.name}/{hour}.0.0"))
            values = np.frombuffer(raw, dtype="<f4").reshape(self.grid)
            kept.append(_quantised(values[self.rows][:, self.columns], field.scale, field.offset))
        return np.stack(kept)

    def hours(
        self, hours: Sequence[int], fields: Sequence[Field], fetch: Fetch, concurrency: int
    ) -> Int16Array:
        """Hours as (step, field, row, column) int16, fetched `concurrency` at a time."""
        with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
            return np.stack(list(pool.map(lambda hour: self.hour(hour, fields, fetch), hours)))


def frames(header: Header, month: Int16Array) -> list[bytes]:
    """A month shaped (step, field, row, column) as its frames, one per day and tile."""
    box = header.box
    made = []
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
                made.append(FRAME_HEADER.pack(day, tile_row, tile_column, 0) + encoded(block))
    return made


def write(path: Path, header: Header, month: Int16Array) -> None:
    """Writes a month shaped (step, field, row, column)."""
    made = frames(header, month)
    start = HEADER.size + OFFSET.size * (len(made) + 1)
    offsets = np.concatenate([[0], np.cumsum([len(frame) for frame in made])]) + start
    with path.open("wb") as file:
        file.write(header.pack())
        file.write(offsets.astype("<u8").tobytes())
        for frame in made:
            file.write(frame)


def read_tile(data: bytes, day: int, tile_row: int, tile_column: int) -> Int16Array:
    """A tile's day from a whole `.surface` file: (step, field, row, column)."""
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
    shape = (STEPS_PER_DAY, len(header.fields), rows, columns)
    return decoded(data[start + FRAME_HEADER.size : end], shape)


def _hours(year: int, month: int, days: int) -> list[int]:
    first = hour_of(datetime(year, month, 1, tzinfo=UTC))
    return list(range(first, first + days * STEPS_PER_DAY))


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
    """Builds `surface-YYYY-MM.surface` from ARCO-ERA5; `days` cuts the month short (pilots)."""
    store = Store.opened(fetch, box, url)
    count = days or calendar.monthrange(year, month)[1]
    values = store.hours(_hours(year, month, count), FIELDS, fetch, concurrency)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"surface-{year}-{month:02d}.surface"
    write(path, Header(year, month, count, box), values)
    logger.info("%s: %d hours, %.0f MB", path.name, len(values), path.stat().st_size / 1e6)
    return path


def measure(
    year: int,
    month: int,
    days: int,
    fetch: Fetch = web.read,
    *,
    box: Box = EUROPE,
    url: str = STORE,
    candidates: Sequence[Field] = CANDIDATES,
    concurrency: int = CONCURRENCY,
) -> list[dict[str, Any]]:
    """What each candidate field would weigh in the archive, from `days` of a month: its
    extremes, and its frames' bytes scaled to a year."""
    store = Store.opened(fetch, box, url)
    hours = _hours(year, month, days)
    rows = []
    for name in dict.fromkeys(field.name for field in candidates):
        same = [field for field in candidates if field.name == name]
        values = store.hours(hours, same, fetch, concurrency)
        for index, field in enumerate(same):
            one = values[:, index : index + 1]
            size = sum(
                len(frame) for frame in frames(Header(year, month, days, box, (field,)), one)
            )
            physical = one.astype(np.float64) * field.scale + field.offset
            rows.append(
                {
                    "field": field.name,
                    "scale": field.scale,
                    "min": float(physical.min()),
                    "max": float(physical.max()),
                    "bytes": size,
                    "gb_per_year": size * 365.25 / days / 1e9,
                }
            )
    return rows


def available(year: int, month: int, fetch: Fetch = web.read, url: str = STORE) -> bool:
    """Whether ARCO-ERA5 holds the month's last hour yet."""
    last = datetime(year, month, calendar.monthrange(year, month)[1], 23, tzinfo=UTC)
    try:
        fetch(f"{url}/{FIELDS[0].name}/{hour_of(last)}.0.0")
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
            if f"surface-{year}-{month:02d}.surface" not in built
            and available(year, month, fetch, url)
        ]
        if months:
            jobs.append({"year": year, "months": " ".join(str(month) for month in months)})
    return jobs
