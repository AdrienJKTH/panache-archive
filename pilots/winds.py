"""Pilot of a wind archive for Panache's back-trajectories (feature 2): measures, builds nothing.

ARCO-ERA5's chunks are whole global hours, so a trajectory read straight from it costs gigabytes.
The candidate archive: u, v and omega on 10 pressure levels (1000-500 hPa) and the surface
pressure, over 15-75° N and 60° W-50° E, every 3 hours, as int16. This pilot reads a few days and
reports:
- how ARCO-ERA5 stores the variables (chunks, codecs) and what reading one hour costs;
- the compressed size of each candidate layout, per year;
- what one 96-hour back-trajectory (an 81-member ensemble, Panache's integrator) would read in
  each layout.
"""

import argparse
import json
import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import gcsfs
import numpy as np
import pyarrow as pa
import xarray as xr
from panache.trajectories.ensemble import Spread, starts
from panache.trajectories.field import WindField
from panache.trajectories.integrate import integrate

STORE = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
WINDS = {"u": "u_component_of_wind", "v": "v_component_of_wind", "omega": "vertical_velocity"}
SURFACE = "surface_pressure"
LEVELS = (500, 600, 700, 800, 850, 900, 925, 950, 975, 1000)
SOUTH, NORTH, WEST, EAST = 15.0, 85.0, -70.0, 90.0
FIRST_BOX = (15.0, 75.0, -60.0, 50.0)
"""The first pilot's box: south, north, west, east."""
STEP_HOURS = 3
STEPS_PER_YEAR = 365.25 * 24 / STEP_HOURS
SCALE = {"u": 0.01, "v": 0.01, "omega": 0.002, "sp": 2.0}
"""Physical units per int16 unit: m/s, m/s, Pa/s, Pa."""
OFFSET = {"u": 0.0, "v": 0.0, "omega": 0.0, "sp": 80_000.0}
TILE = 20
"""Grid points per tile side at 0.5°: 10°."""
PLACES = {
    "Lyon": (45.76, 4.84),
    "Paris": (48.86, 2.35),
    "Bordeaux": (44.84, -0.58),
    "Warsaw": (52.23, 21.01),
    "Athens": (37.98, 23.73),
    "Helsinki": (60.17, 24.94),
}


@dataclass
class Step:
    """One time step over the box at 0.25°: winds (field, level, lat, lon), surface (lat, lon)."""

    time: datetime
    winds: np.ndarray
    surface: np.ndarray
    seconds: float


def metadata(fs: gcsfs.GCSFileSystem, variable: str) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads(fs.cat(f"{STORE}/{variable}/zarr.json"))
    return meta


def chunk_report(fs: gcsfs.GCSFileSystem, variable: str, time_index: int) -> dict[str, Any]:
    """The variable's chunk grid and codecs, and the stored size of the chunks of one hour."""
    meta = metadata(fs, variable)
    shape = meta["shape"]
    chunk = meta["chunk_grid"]["configuration"]["chunk_shape"]
    separator = meta.get("chunk_key_encoding", {}).get("configuration", {}).get("separator", "/")
    counts = [math.ceil(s / c) for s, c in zip(shape, chunk, strict=True)]
    report: dict[str, Any] = {
        "shape": shape,
        "chunk": chunk,
        "dimensions": meta.get("dimension_names"),
        "codecs": [codec["name"] for codec in meta.get("codecs", [])],
        "data_type": meta.get("data_type"),
    }
    first = time_index // chunk[0]
    keys = [[first]]
    for count in counts[1:]:
        keys = [[*key, index] for key in keys for index in range(count)]
    sizes = []
    for key in keys[:64]:
        path = f"{STORE}/{variable}/c{separator}" + separator.join(str(k) for k in key)
        try:
            sizes.append(int(fs.info(path)["size"]))
        except FileNotFoundError:
            sizes.append(0)
    report["chunks_per_hour"] = len(keys)
    report["bytes_per_hour"] = sum(sizes) * len(keys) / len(sizes)
    return report


def box_indices(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat = ds.latitude.values
    lon = ds.longitude.values
    lat_index = np.flatnonzero((lat >= SOUTH - 1e-6) & (lat <= NORTH + 1e-6))
    lat_index = lat_index[np.argsort(lat[lat_index])]
    signed = np.where(lon > 180, lon - 360, lon)
    lon_index = np.flatnonzero((signed >= WEST - 1e-6) & (signed <= EAST + 1e-6))
    lon_index = lon_index[np.argsort(signed[lon_index])]
    return lat_index, lon_index, lat[lat_index], signed[lon_index]


def read_step(ds: xr.Dataset, index: int, lats: np.ndarray, lons: np.ndarray) -> Step:
    started = time.perf_counter()
    region = {"time": index, "latitude": lats, "longitude": lons}
    winds = np.stack(
        [
            ds[name].isel(region).sel(level=list(LEVELS)).values.astype(np.float32)
            for name in WINDS.values()
        ]
    )
    surface = ds[SURFACE].isel(region).values.astype(np.float32)
    moment = datetime.fromisoformat(str(ds.time.values[index])[:19]).replace(tzinfo=UTC)
    return Step(moment, winds, surface, time.perf_counter() - started)


def quantised(step: Step, every: int) -> tuple[np.ndarray, dict[str, int]]:
    """The step as int16 (31 fields: u, v, omega on 10 levels, then sp), and values clipped."""
    fields = []
    clipped = {}
    for name, values in (
        ("u", step.winds[0]),
        ("v", step.winds[1]),
        ("omega", step.winds[2]),
        ("sp", step.surface[np.newaxis]),
    ):
        scaled = np.round((values[:, ::every, ::every] - OFFSET[name]) / SCALE[name])
        clipped[name] = int((np.abs(scaled) > 32767).sum())
        fields.append(np.clip(scaled, -32767, 32767).astype(np.int16))
    return np.concatenate(fields), clipped


def dequantised(stack: np.ndarray) -> dict[str, np.ndarray]:
    k = len(LEVELS)
    return {
        "u": stack[..., :k, :, :] * SCALE["u"],
        "v": stack[..., k : 2 * k, :, :] * SCALE["v"],
        "omega": stack[..., 2 * k : 3 * k, :, :] * SCALE["omega"],
        "sp": stack[..., 3 * k, :, :] * SCALE["sp"] + OFFSET["sp"],
    }


def shuffled(values: np.ndarray) -> bytes:
    """The bytes of an int16 array, low bytes first, then high bytes (byte stream split)."""
    return np.ascontiguousarray(values).view(np.uint8).reshape(-1, 2).T.tobytes()


def compressed(data: bytes, level: int = 3) -> int:
    codec = pa.Codec("zstd", compression_level=level)
    return len(codec.compress(data, asbytes=True))


ENCODINGS: dict[str, Callable[[np.ndarray], int]] = {
    "plain, zstd 3": lambda a: compressed(a.tobytes()),
    "byte split, zstd 3": lambda a: compressed(shuffled(a)),
    "byte split, zstd 19": lambda a: compressed(shuffled(a), 19),
    "delta along lon, byte split, zstd 3": lambda a: compressed(
        shuffled(np.diff(a, axis=-1, prepend=np.zeros_like(a[..., :1])))
    ),
}


def tiles(shape: tuple[int, int]) -> list[tuple[int, int]]:
    return [(j, i) for j in range(0, shape[0], TILE) for i in range(0, shape[1], TILE)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2021-02-02")
    parser.add_argument("--days", type=int, default=6)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    lines = ["# Winds pilot", ""]

    fs = gcsfs.GCSFileSystem(token="anon")
    opened = time.perf_counter()
    ds = xr.open_zarr(f"gs://{STORE}", storage_options={"token": "anon"}, chunks=None)
    result["open_seconds"] = time.perf_counter() - opened
    result["time_range"] = [str(ds.time.values[0]), str(ds.time.values[-1])]
    lines.append(f"ARCO-ERA5 opened in {result['open_seconds']:.1f} s, {result['time_range']}.")

    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    first = int(np.searchsorted(ds.time.values, np.datetime64(start.replace(tzinfo=None))))
    indices = list(range(first, first + args.days * 24, STEP_HOURS))
    result["chunks"] = {}
    for variable in (*WINDS.values(), SURFACE):
        try:
            result["chunks"][variable] = chunk_report(fs, variable, first)
        except Exception as error:
            listing = {}
            for path in (STORE, f"{STORE}/{variable}"):
                try:
                    listing[path] = [entry["name"] for entry in fs.ls(path, detail=True)[:12]]
                except Exception as inner:
                    listing[path] = [repr(inner)]
            encoding = {key: str(value) for key, value in ds[variable].encoding.items()}
            result["chunks"][variable] = {
                "error": repr(error),
                "listing": listing,
                "encoding": encoding,
            }
        lines.append(f"- `{variable}`: {result['chunks'][variable]}")

    lat_index, lon_index, lats, lons = box_indices(ds)
    began = time.perf_counter()
    with ThreadPoolExecutor(args.threads) as pool:
        steps = list(pool.map(lambda i: read_step(ds, i, lat_index, lon_index), indices))
    wall = time.perf_counter() - began
    per_step = [step.seconds for step in steps]
    result["read"] = {
        "steps": len(steps),
        "threads": args.threads,
        "wall_seconds": wall,
        "wall_per_step": wall / len(steps),
        "step_seconds_median": float(np.median(per_step)),
        "step_seconds_max": float(np.max(per_step)),
    }
    hours_per_year_of_runner = wall / len(steps) * STEPS_PER_YEAR / 3600
    result["read"]["runner_hours_per_year"] = hours_per_year_of_runner
    lines += [
        "",
        f"Read {len(steps)} steps on {args.threads} threads in {wall:.0f} s: "
        f"{wall / len(steps):.2f} s a step, ~{hours_per_year_of_runner:.1f} runner hours a year.",
        f"Box at 0.25°: {lats.size} by {lons.size}; latitudes {lats[0]}-{lats[-1]}, "
        f"longitudes {lons[0]}-{lons[-1]}.",
        f"Ranges: u {float(np.abs(np.stack([s.winds[0] for s in steps])).max()):.1f} m/s, "
        f"omega {float(np.abs(np.stack([s.winds[2] for s in steps])).max()):.2f} Pa/s, "
        f"sp {float(min(s.surface.min() for s in steps)):.0f}-"
        f"{float(max(s.surface.max() for s in steps)):.0f} Pa.",
    ]

    sizes: dict[str, float] = {}
    clipped_total: dict[str, int] = {}
    stacks: dict[int, list[np.ndarray]] = {1: [], 2: []}
    for every in (1, 2):
        for step in steps:
            stack, clipped = quantised(step, every)
            stacks[every].append(stack)
            for name, count in clipped.items():
                clipped_total[name] = clipped_total.get(name, 0) + count
    result["clipped"] = clipped_total
    lines += ["", f"Values clipped to int16: {clipped_total}", "", "## Sizes per year", ""]
    lines += ["| Layout | Per frame | Per year |", "|---|---|---|"]

    def record(name: str, per_frame: float, frames_per_year: float) -> None:
        sizes[name] = per_frame * frames_per_year
        lines.append(f"| {name} | {per_frame / 1e3:.0f} kB | {sizes[name] / 1e9:.2f} GB |")

    raw = stacks[2][0].nbytes
    record("0.5°, raw int16, a frame per step", raw, STEPS_PER_YEAR)
    for name, encode in ENCODINGS.items():
        started = time.perf_counter()
        mean = float(np.mean([encode(stack) for stack in stacks[2]]))
        record(f"0.5°, {name}, a frame per step", mean, STEPS_PER_YEAR)
        result.setdefault("encode_seconds_per_step", {})[name] = (
            time.perf_counter() - started
        ) / len(stacks[2])
    best = ENCODINGS["delta along lon, byte split, zstd 3"]
    record(
        "0.25°, delta along lon, byte split, zstd 3, a frame per step",
        float(np.mean([best(stack) for stack in stacks[1]])),
        STEPS_PER_YEAR,
    )
    per_day = 24 // STEP_HOURS
    days = [np.stack(stacks[2][d : d + per_day]) for d in range(0, len(stacks[2]), per_day)]
    record(
        "0.5°, delta along lon, byte split, zstd 3, a frame per day",
        float(np.mean([best(day) for day in days])),
        365.25,
    )
    record(
        "0.5°, delta in time, byte split, zstd 3, a frame per day",
        float(
            np.mean(
                [
                    compressed(shuffled(np.diff(day, axis=0, prepend=np.zeros_like(day[:1]))))
                    for day in days
                ]
            )
        ),
        365.25,
    )
    grid = stacks[2][0].shape[1:]
    tile_sizes = {
        (d, tile): best(day[:, :, tile[0] : tile[0] + TILE, tile[1] : tile[1] + TILE])
        for d, day in enumerate(days)
        for tile in tiles(grid)
    }
    record(
        f"0.5°, delta along lon, byte split, zstd 3, a frame per day and {TILE // 2}° tile",
        float(np.mean(list(tile_sizes.values()))) * len(tiles(grid)),
        365.25,
    )
    coarse = [stack.copy() for stack in stacks[2]]
    for stack in coarse:
        winds = stack[: 2 * len(LEVELS)].astype(np.int32)
        stack[: 2 * len(LEVELS)] = np.round(winds / 10).astype(np.int16)
    coarse_days = [np.stack(coarse[d : d + per_day]) for d in range(0, len(coarse), per_day)]
    record(
        f"0.5°, u and v to 0.1 m/s, delta along lon, byte split, zstd 3, a frame per day and "
        f"{TILE // 2}° tile",
        float(
            np.mean(
                [
                    best(day[:, :, tile[0] : tile[0] + TILE, tile[1] : tile[1] + TILE])
                    for day in coarse_days
                    for tile in tiles(grid)
                ]
            )
        )
        * len(tiles(grid)),
        365.25,
    )
    result["sizes_per_year"] = sizes

    # The field at 0.5°, from the int16 values, and real back-trajectories in it.
    whole = dequantised(np.stack(stacks[2]).astype(np.float64))
    field = WindField(
        times=np.array([step.time.timestamp() for step in steps]),
        levels=np.array(LEVELS, dtype=np.float64) * 100,
        lats=lats[::2].astype(np.float64),
        lons=lons[::2].astype(np.float64),
        u=whole["u"],
        v=whole["v"],
        omega=whole["omega"],
        surface_pressure=whole["sp"],
    )
    step_frame = float(np.mean([best(stack) for stack in stacks[2]]))
    day_frame = sizes["0.5°, delta along lon, byte split, zstd 3, a frame per day"] / 365.25
    moment = start + timedelta(days=4, hours=12)
    lines += ["", f"## 96-hour back-trajectories from {moment:%Y-%m-%d %H} UTC, 500 m, 81 members"]
    lines += [
        "",
        "| Place | Centre after 96 h | Outside | Left the first box | Reach (S, N, W, E) |"
        " Integration | Frames per step | Per day | Tiles per day |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    result["trajectories"] = {}
    for place, (lat, lon) in PLACES.items():
        particles = starts(field, lat=lat, lon=lon, metres=500, moment=moment, spread=Spread())
        began = time.perf_counter()
        paths = integrate(field, particles, hours=96, backward=True)
        seconds = time.perf_counter() - began
        offsets = (paths.time - field.times[0]) / (STEP_HOURS * 3600)
        steps_read = set(np.floor(offsets).astype(int).ravel()) | set(
            np.ceil(offsets).astype(int).ravel()
        )
        days_read = {s // per_day for s in steps_read}
        tiles_read: set[tuple[int, tuple[int, int]]] = set()
        valid = ~np.isnan(paths.lat)
        rows = (paths.lat[valid] - field.lats[0]) / 0.5
        columns = (paths.lon[valid] - field.lons[0]) / 0.5
        for step_offset, row, column in zip(offsets[valid], rows, columns, strict=True):
            for s in {math.floor(step_offset), math.ceil(step_offset)}:
                for r in {math.floor(row), math.ceil(row)}:
                    for c in {math.floor(column), math.ceil(column)}:
                        r_, c_ = min(r, grid[0] - 1), min(c, grid[1] - 1)
                        tiles_read.add((s // per_day, (r_ // TILE * TILE, c_ // TILE * TILE)))
        tile_bytes = sum(tile_sizes.get(key, 0) for key in tiles_read)
        centre = (float(paths.lat[0, -1]), float(paths.lon[0, -1]))
        south, north, west, east = FIRST_BOX
        left_first_box = (
            (paths.lat < south) | (paths.lat > north) | (paths.lon < west) | (paths.lon > east)
        ).any(axis=1)
        reach = [
            float(np.nanmin(paths.lat)),
            float(np.nanmax(paths.lat)),
            float(np.nanmin(paths.lon)),
            float(np.nanmax(paths.lon)),
        ]
        entry = {
            "centre_end": centre,
            "reach_south_north_west_east": reach,
            "left_first_box_share": float(left_first_box.mean()),
            "outside_share": float(paths.outside[:, -1].mean()),
            "grounded_share": float(paths.grounded.mean()),
            "integration_seconds": seconds,
            "step_frames": len(steps_read),
            "step_bytes": len(steps_read) * step_frame,
            "day_frames": len(days_read),
            "day_bytes": len(days_read) * day_frame,
            "tile_frames": len(tiles_read),
            "tile_bytes": tile_bytes,
        }
        result["trajectories"][place] = entry
        lines.append(
            f"| {place} | {centre[0]:.1f}° N {centre[1]:.1f}° E | "
            f"{entry['outside_share']:.0%} | {entry['left_first_box_share']:.0%} | "
            f"{', '.join(f'{value:.0f}' for value in reach)} | {seconds:.2f} s | "
            f"{len(steps_read)}, {entry['step_bytes'] / 1e6:.1f} MB | "
            f"{len(days_read)}, {entry['day_bytes'] / 1e6:.1f} MB | "
            f"{len(tiles_read)}, {tile_bytes / 1e6:.1f} MB |"
        )

    (args.out / "winds-pilot.json").write_text(json.dumps(result, indent=2, default=str))
    (args.out / "winds-pilot.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
