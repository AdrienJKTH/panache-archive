"""CAMS's global anthropogenic emissions for Panache's Lagrangian footprints (Panache's D86).

A footprint says how much a surface flux in each cell raises a receptor's concentration; times
an emission inventory, it says which country and which sector emitted what a place breathes.

CAMS-GLOB-ANT (Soulie et al. 2024, ESSD 16, 2261-2279, doi:10.5194/essd-16-2261-2024) gives
monthly emissions by sector at 0.1°, 2000 onwards. It is read from the Atmosphere Data Store's
`cams-global-emission-inventories` (licence CC-BY-4.0, read on 2026-11-13 in the store's
catalogue), with the key the CAMS reanalysis builds use.

It has **no PM2.5**: its particles are black carbon and organic carbon. With nitrogen oxides
those are the species taken (`SPECIES`).

- **Content**: each sector's monthly emission, kg m⁻² s⁻¹, over the winds archive's box
  (15-85° N, 70° W-90° E) in the inventory's own 0.1° cells: 700 rows by 1,600 columns. The
  inventory's `sum` is left out; a build checks that the sectors add up to it.
- **Values**: float16 of the emission times `SCALE` (1e11): three significant digits, from
  6e-16 to 6.5e-7 kg m⁻² s⁻¹. A build stops if a cell is beyond.
- **Assets**: one per species and year, `emissions-black_carbon-2021.emissions` (~30 MB;
  measured by the pilot, panache-archive#55).

The format, little-endian, after the `.winds` files':
- **Header**, 64 bytes (`HEADER`): magic, version, year, months, sectors, rows, columns, tile
  side, then as float64 the south and west centres, the step and the scale.
- **Sector names**: 8 ASCII bytes each, padded with zeros (`res`, `tro`, ...).
- **Offset table**: one `uint64` per frame and one more. The frame of month `m` (from 0) and
  tile `(r, c)` is `(m * tile_rows + r) * tile_columns + c`.
- **Frames**: 8 bytes saying which month and tile follow, then zstd over the float16 values
  shaped (sector, row, column), byte split (low bytes, then high bytes).
"""

import logging
import struct
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import cdsapi
import netCDF4
import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from panache_archive import ads

type FloatArray = NDArray[np.float64]
type Float16Array = NDArray[np.float16]
Download = Callable[[dict[str, list[str]], Path], None]

logger = logging.getLogger(__name__)

DATASET: Final = "cams-global-emission-inventories"
VERSION: Final = "v6.2"
SPECIES: Final = ("black_carbon", "organic_carbon", "nitrogen_oxides")
TOTAL: Final = "sum"
"""The inventory's own sum of its sectors: checked, not kept."""
SCALE: Final = 1e11
MONTHS: Final = 12
FIRST_YEAR: Final = 2013
LAST_YEAR: Final = 2026
"""The last year of v6.2 in the store's catalogue."""
EARTH_RADIUS_M: Final = 6_371_000.0
ZSTD_LEVEL: Final = 3
SECONDS_PER_YEAR: Final = 365.25 * 86_400
MAGIC: Final = b"PANEMISS"
FORMAT: Final = 1
HEADER: Final = struct.Struct("<8s7H4d10x")
NAME: Final = struct.Struct("<8s")
FRAME_HEADER: Final = struct.Struct("<4H")
"""Month, tile row, tile column, and zero."""
OFFSET: Final = struct.Struct("<Q")


@dataclass(frozen=True, slots=True)
class Box:
    """Cells of `step` degrees whose centres start at (south, west), cut in square tiles."""

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


EUROPE: Final = Box(south=15.05, west=-69.95, step=0.1, rows=700, columns=1600, tile=100)
"""The winds archive's box (15-85° N, 70° W-90° E) in the inventory's 0.1° cells, 10° tiles."""


@dataclass(frozen=True, slots=True)
class Header:
    year: int
    sectors: tuple[str, ...]
    box: Box

    @property
    def size(self) -> int:
        """The bytes before the offset table."""
        return HEADER.size + NAME.size * len(self.sectors)

    def frame(self, month: int, tile_row: int, tile_column: int) -> int:
        return (month * self.box.tile_rows + tile_row) * self.box.tile_columns + tile_column

    def pack(self) -> bytes:
        box = self.box
        head = HEADER.pack(
            MAGIC,
            FORMAT,
            self.year,
            MONTHS,
            len(self.sectors),
            box.rows,
            box.columns,
            box.tile,
            box.south,
            box.west,
            box.step,
            SCALE,
        )
        return head + b"".join(NAME.pack(name.encode("ascii")) for name in self.sectors)

    @classmethod
    def unpack(cls, data: bytes) -> "Header":
        magic, version, year, months, count, rows, columns, tile, south, west, step, scale = (
            HEADER.unpack(data[: HEADER.size])
        )
        if (magic, version, months, scale) != (MAGIC, FORMAT, MONTHS, SCALE):
            raise ValueError("not a version 1 `.emissions` file of this layout")
        names = tuple(
            NAME.unpack_from(data, HEADER.size + NAME.size * index)[0].rstrip(b"\0").decode("ascii")
            for index in range(count)
        )
        return cls(year, names, Box(south, west, step, rows, columns, tile))


def request(species: str, year: int) -> dict[str, list[str]]:
    return {
        "variable": [species],
        "source": ["anthropogenic"],
        "version": [VERSION],
        "year": [str(year)],
    }


def download(wanted: dict[str, list[str]], target: Path) -> None:
    """Fetch with the ADS key of `CDSAPI_URL` and `CDSAPI_KEY`, or of `~/.cdsapirc`."""
    client = cdsapi.Client(quiet=True, progress=False)
    client.retrieve(DATASET, dict(wanted), str(target))


def _positions(axis: FloatArray, wanted: FloatArray) -> NDArray[np.intp]:
    """Where each wanted centre lies on the file's axis, which must hold it."""
    index = np.abs(axis[:, np.newaxis] - wanted[np.newaxis, :]).argmin(axis=0)
    if np.abs(axis[index] - wanted).max() > 1e-3:
        raise ValueError("the inventory's grid does not hold the archive's cells")
    return index.astype(np.intp)


def _areas(lats: FloatArray, step: float) -> FloatArray:
    """Each row's cell area, m²."""
    edges = np.radians(np.stack([lats - step / 2, lats + step / 2]))
    band: FloatArray = EARTH_RADIUS_M**2 * np.radians(step) * (np.sin(edges[1]) - np.sin(edges[0]))
    return band


def _tiles(values: NDArray[Any], tile: int) -> Iterator[NDArray[Any]]:
    """An array whose last axes are (row, column) as its square tiles, row by row."""
    for row in range(0, values.shape[-2], tile):
        for column in range(0, values.shape[-1], tile):
            yield values[..., row : row + tile, column : column + tile]


def encoded(block: NDArray[Any]) -> bytes:
    """A block's values, byte split (low bytes, then high bytes) and compressed."""
    raw = np.ascontiguousarray(block)
    split = raw.view(np.uint8).reshape(-1, raw.dtype.itemsize).T.tobytes()
    return bytes(pa.Codec("zstd", compression_level=ZSTD_LEVEL).compress(split, asbytes=True))


def decoded(data: bytes, shape: tuple[int, int, int]) -> Float16Array:
    raw = pa.Codec("zstd").decompress(data, decompressed_size=2 * int(np.prod(shape)), asbytes=True)
    return np.frombuffer(raw, dtype=np.uint8).reshape(2, -1).T.copy().view("<f2").reshape(shape)


def _axes(dataset: netCDF4.Dataset) -> tuple[str, str]:
    """The names of the file's latitude and longitude."""
    lat, lon = (
        next(name for name in names if name in dataset.variables)
        for names in (("lat", "latitude"), ("lon", "longitude"))
    )
    return lat, lon


def read(path: Path, box: Box = EUROPE) -> tuple[tuple[str, ...], NDArray[np.float32], FloatArray]:
    """A year's file over the box: the sectors' names, their emissions shaped (month, sector,
    row, column) in kg m⁻² s⁻¹, and the inventory's own sum shaped (month, row, column)."""
    with netCDF4.Dataset(path) as dataset:
        lat, lon = _axes(dataset)
        rows = _positions(np.asarray(dataset.variables[lat][:], dtype=np.float64), box.lats)
        columns = _positions(np.asarray(dataset.variables[lon][:], dtype=np.float64), box.lons)
        fields = {
            name: np.ma.filled(variable[:, rows, columns], 0.0).astype(np.float32)
            for name, variable in dataset.variables.items()
            if variable.dimensions == ("time", lat, lon)
        }
    total = fields.pop(TOTAL)
    if not fields or total.shape[0] != MONTHS:
        raise ValueError(f"{path.name} does not hold {MONTHS} months of sectors")
    names = tuple(fields)
    return names, np.stack([fields[name] for name in names], axis=1), total.astype(np.float64)


def write(path: Path, header: Header, year: NDArray[np.float32]) -> None:
    """Writes a year shaped (month, sector, row, column), kg m⁻² s⁻¹."""
    scaled = year.astype(np.float64) * SCALE
    if not np.isfinite(scaled).all() or scaled.min() < 0 or scaled.max() > np.finfo(np.float16).max:
        raise ValueError("emissions missing, negative or beyond float16")
    kept = scaled.astype("<f2")
    box = header.box
    frames = []
    for month in range(MONTHS):
        for index, block in enumerate(_tiles(kept[month], box.tile)):
            tile_row, tile_column = divmod(index, box.tile_columns)
            frames.append(FRAME_HEADER.pack(month, tile_row, tile_column, 0) + encoded(block))
    start = header.size + OFFSET.size * (len(frames) + 1)
    offsets = np.concatenate([[0], np.cumsum([len(frame) for frame in frames])]) + start
    with path.open("wb") as file:
        file.write(header.pack())
        file.write(offsets.astype("<u8").tobytes())
        for frame in frames:
            file.write(frame)


def read_tile(data: bytes, month: int, tile_row: int, tile_column: int) -> FloatArray:
    """A tile's month from a whole `.emissions` file: (sector, row, column), kg m⁻² s⁻¹."""
    header = Header.unpack(data)
    box = header.box
    index = header.frame(month, tile_row, tile_column)
    start, end = (
        OFFSET.unpack_from(data, header.size + OFFSET.size * i)[0] for i in (index, index + 1)
    )
    if FRAME_HEADER.unpack_from(data, start) != (month, tile_row, tile_column, 0):
        raise ValueError(f"frame {index} is not month {month}, tile ({tile_row}, {tile_column})")
    rows = min(box.tile, box.rows - tile_row * box.tile)
    columns = min(box.tile, box.columns - tile_column * box.tile)
    shape = (len(header.sectors), rows, columns)
    return decoded(data[start + FRAME_HEADER.size : end], shape).astype(np.float64) / SCALE


def _netcdf(species: str, year: int, workdir: Path, fetch: Download) -> tuple[Path, int]:
    """The species' year downloaded: its netCDF file, and the bytes that came."""
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / f"{species}-{year}.download"
    fetch(request(species, year), target)
    size = target.stat().st_size
    files = ads.netcdf_files(target, workdir)
    if len(files) != 1:
        raise ValueError(f"expected one netCDF file for {species} in {year}, got {len(files)}")
    return files[0], size


def build(
    species: str,
    year: int,
    out_dir: Path,
    workdir: Path,
    fetch: Download = download,
    *,
    box: Box = EUROPE,
) -> Path:
    """Builds `emissions-<species>-<year>.emissions` from the store."""
    source, _ = _netcdf(species, year, workdir, fetch)
    try:
        sectors, values, total = read(source, box)
    finally:
        source.unlink()
    summed = values.sum(axis=1, dtype=np.float64)
    # No absolute tolerance: these are sums of kg m⁻² s⁻¹, far below numpy's default of 1e-8.
    if not np.isclose(summed.sum(), total.sum(), rtol=0.01, atol=0.0):
        raise ValueError(
            f"{species} {year}: the sectors add up to {summed.sum():.4g}, "
            f"the inventory's sum to {total.sum():.4g}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"emissions-{species}-{year}.emissions"
    write(path, Header(year, sectors, box), values)
    areas = _areas(box.lats, box.step)[:, np.newaxis]
    yearly = float((summed.mean(axis=0) * areas).sum()) * SECONDS_PER_YEAR / 1e6
    logger.info(
        "%s: %s, %.0f kt a year over the box, %.1f MB",
        path.name,
        " ".join(sectors),
        yearly,
        path.stat().st_size / 1e6,
    )
    return path


def plan(years: Sequence[int], built: set[str]) -> list[dict[str, Any]]:
    """The matrix: each species and year not built yet."""
    return [
        {"species": species, "year": year}
        for species in SPECIES
        for year in years
        if f"emissions-{species}-{year}.emissions" not in built
    ]


def describe(path: Path, box: Box = EUROPE) -> list[str]:
    """A netCDF file of the inventory as markdown lines: its axes, then each field over the
    box (its total, the share of cells that emit, and what it would weigh)."""
    lines = [f"**{path.name}**, {path.stat().st_size / 1e6:.1f} MB"]
    with netCDF4.Dataset(path) as dataset:
        dims = ", ".join(f"{name} {len(dim)}" for name, dim in dataset.dimensions.items())
        lines.append(f"- dimensions: {dims}")
        lat, lon = _axes(dataset)
        for name in (lat, lon, "time"):
            axis = dataset.variables[name]
            values = np.asarray(axis[:], dtype=np.float64)
            lines.append(f"- {name}: {values[0]:g} to {values[-1]:g}, {getattr(axis, 'units', '')}")
        rows = _positions(np.asarray(dataset.variables[lat][:], dtype=np.float64), box.lats)
        columns = _positions(np.asarray(dataset.variables[lon][:], dtype=np.float64), box.lons)
        areas = _areas(box.lats, box.step)[:, np.newaxis]
        lines += [
            "",
            "| Field | Long name | Unit | Box, kt a year | Cells emitting | float32, MB |"
            " float16, MB |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, variable in dataset.variables.items():
            if variable.dimensions != ("time", lat, lon):
                continue
            cut = np.ma.filled(variable[:, rows, columns], 0.0).astype(np.float32)
            yearly = float((cut.mean(axis=0) * areas).sum()) * SECONDS_PER_YEAR / 1e6
            small = (cut * SCALE).astype(np.float16)
            lines.append(
                f"| {name} | {getattr(variable, 'long_name', '')} "
                f"| {getattr(variable, 'units', '')} | {yearly:.1f} "
                f"| {np.count_nonzero(cut.max(axis=0)) / cut[0].size:.1%} "
                f"| {sum(len(encoded(tile)) for tile in _tiles(cut, box.tile)) / 1e6:.2f} "
                f"| {sum(len(encoded(tile)) for tile in _tiles(small, box.tile)) / 1e6:.2f} |"
            )
    return lines


def pilot(
    species: str, year: int, workdir: Path, fetch: Download = download, box: Box = EUROPE
) -> list[str]:
    """Downloads a species' year and describes the file that came."""
    source, size = _netcdf(species, year, workdir, fetch)
    lines = [f"### {species}, {year}, {VERSION}: {size / 1e6:.1f} MB downloaded", ""]
    lines += [*describe(source, box), ""]
    source.unlink()
    return lines
