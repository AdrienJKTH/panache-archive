# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "eccodes>=2.41",
#     "httpx>=0.28",
#     "numpy>=2.0",
#     "pyarrow>=20.0",
# ]
# ///
"""Panache's P6, E1: AIFS Single's 00 UTC runs at a few places, from ECMWF's open data on AWS.

For each run of a month, each step (6-hourly to J+8) and each of eight surface fields, the GRIB
message is fetched by its range (the step's `.index` says where it is), decoded, and the value at
the nearest 0.25° point of each place kept, in the units GRIB declares. Panache turns them into
daily weather (`panache.forecast.aifs`). A run missing a file or a field is left out whole.

The data: "ECMWF open data", CC BY 4.0, modified (values at points).
Run as `uv run --script experiments/aifs_points.py --help`: ecCodes stays out of the project.
"""

import argparse
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import eccodes
import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MIRROR = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
STEPS = range(0, 193, 6)
PARAMS = ("2t", "2d", "10u", "10v", "msl", "tp", "ssrd", "tcc")
ATTEMPTS = 8
CONCURRENCY = 8
"""The mirror answers 503 when asked too fast: few requests at once, and patient retries."""

logger = logging.getLogger("aifs_points")


class MissingError(Exception):
    """The run lacks a file or a field."""


def get(http: httpx.Client, url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {}
    if byte_range is not None:
        offset, length = byte_range
        headers["Range"] = f"bytes={offset}-{offset + length - 1}"
    for attempt in range(ATTEMPTS):
        try:
            response = http.get(url, headers=headers)
            if response.status_code == 404:
                raise MissingError(url)
            response.raise_for_status()
            if byte_range is not None and len(response.content) != byte_range[1]:
                raise httpx.TransportError(f"{url}: {len(response.content)} bytes")
            return response.content
        except httpx.HTTPError as error:
            if attempt == ATTEMPTS - 1:
                raise
            logger.warning("%s, trying again: %s", url, error)
            time.sleep(2**attempt + random.random())
    raise AssertionError("unreachable")


def nearest_points(message: int, places: list[dict]) -> list[int]:
    """Each place's index in the message's values: the nearest point of the regular grid."""
    columns = eccodes.codes_get(message, "Ni")
    latitudes = eccodes.codes_get_array(message, "latitudes")
    longitudes = eccodes.codes_get_array(message, "longitudes")
    out = []
    for place in places:
        east = (longitudes - place["lon"] + 180) % 360 - 180
        index = int(np.argmin((latitudes - place["lat"]) ** 2 + east**2))
        if abs(latitudes[index] - place["lat"]) > 0.126 or abs(east[index]) > 0.126:
            raise ValueError(f"{place['name']} is off the grid of {columns} columns")
        out.append(index)
    return out


class Grid:
    """The places' indices, found once: every file is on the same grid, checked by its shape."""

    def __init__(self, places: list[dict]) -> None:
        self.places = places
        self.shape: tuple = ()
        self.indices: list[int] = []

    def values(self, raw: bytes) -> tuple[str, np.ndarray]:
        message = eccodes.codes_new_from_message(raw)
        try:
            shape = tuple(
                eccodes.codes_get(message, key)
                for key in (
                    "Ni",
                    "Nj",
                    "latitudeOfFirstGridPointInDegrees",
                    "longitudeOfFirstGridPointInDegrees",
                    "jScansPositively",
                    "iScansNegatively",
                )
            )
            if shape != self.shape:
                logger.info("grid: %s", shape)
                self.indices = nearest_points(message, self.places)
                self.shape = shape
            field = eccodes.codes_get_values(message)
            return eccodes.codes_get(message, "units"), field[self.indices]
        finally:
            eccodes.codes_release(message)


def fetch_step(http: httpx.Client, day: date, step: int) -> dict[str, bytes]:
    stem = f"{MIRROR}/{day:%Y%m%d}/00z/aifs-single/0p25/oper/{day:%Y%m%d}000000-{step}h-oper-fc"
    entries = [json.loads(line) for line in get(http, f"{stem}.index").splitlines() if line]
    where = {
        e["param"]: (int(e["_offset"]), int(e["_length"]))
        for e in entries
        if e["levtype"] == "sfc" and e["param"] in PARAMS
    }
    missing = [p for p in PARAMS if p not in where]
    if missing:
        raise MissingError(f"{stem} lacks {missing}")
    return {p: get(http, f"{stem}.grib2", where[p]) for p in PARAMS}


def run_rows(pool: ThreadPoolExecutor, http: httpx.Client, grid: Grid, day: date) -> list[dict]:
    rows = []
    # Decoded here, one at a time: ecCodes is not told to be shared between threads.
    for step, messages in zip(
        STEPS, pool.map(lambda s: fetch_step(http, day, s), STEPS), strict=True
    ):
        for param, raw in messages.items():
            units, values = grid.values(raw)
            rows += [
                {
                    "run": day,
                    "step": step,
                    "param": param,
                    "units": units,
                    "place": place["name"],
                    "value": float(value),
                }
                for place, value in zip(grid.places, values, strict=True)
            ]
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=date.fromisoformat, required=True, help="first run's day")
    parser.add_argument("--last", type=date.fromisoformat, required=True, help="last run's day")
    parser.add_argument("--places", type=Path, required=True, help="JSON: name, lat, lon")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    grid = Grid(json.loads(args.places.read_text(encoding="utf-8")))
    days = [args.first + timedelta(i) for i in range((args.last - args.first).days + 1)]
    rows: list[dict] = []
    left_out = []
    with httpx.Client(timeout=120) as http, ThreadPoolExecutor(CONCURRENCY) as pool:
        for day in days:
            try:
                rows += run_rows(pool, http, grid, day)
            except MissingError as error:
                logger.warning("run of %s left out: %s", day, error)
                left_out.append(day)
            else:
                logger.info("run of %s read", day)
    if not rows:
        raise SystemExit(f"no run between {args.first} and {args.last}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.out, compression="zstd")
    print(f"{len(days) - len(left_out)} runs read, {len(left_out)} left out: {left_out}")


if __name__ == "__main__":
    main()
