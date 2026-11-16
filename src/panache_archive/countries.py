"""Each 0.1° cell's country, for the shares of a footprint's emissions by country (Panache's D86).

The cells are the emission inventories' (`emissions.EUROPE`: the winds archive's box, 15-85° N,
70° W-90° E, 700 rows by 1,600 columns): a footprint times an inventory, summed over a
country's cells, is that country's share.

- **Source**: Natural Earth's admin 0 countries at 1:10 million, v5.1.2 (public domain), burnt
  by GDAL (`scripts/countries.sh`) with each country's `NE_ID` on cells ten times finer.
- **A cell's country** is the one holding most of its land, among the 100 finer cells it
  covers. A cell without land is the sea's (code 0): ships' emissions, and a coast's cells
  whose land Natural Earth misses.
- **Asset**: `countries.npz` in the release `emissions-v1`: `code` (uint16, rows from the
  south, columns from the west), `iso` and `name` (index 0 the sea's), and the grid's `south`,
  `west` (cell centres) and `step`.
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
import tifffile
from numpy.typing import NDArray

from panache_archive.emissions import EUROPE, Box

FINER: Final = 10
"""Finer cells per cell, along each side."""
SEA: Final = ("", "sea")
"""Code 0's ISO code and name."""
ASSET: Final = "countries.npz"
SOURCE: Final = "Natural Earth 1:10m admin 0 countries, v5.1.2 (public domain)"


def read_countries(path: Path) -> dict[int, tuple[str, str]]:
    """Each country's `NE_ID`, with its `ADM0_A3` code and its name."""
    found: dict[int, tuple[str, str]] = {}
    for feature in json.loads(path.read_text(encoding="utf-8"))["features"]:
        properties = feature["properties"]
        found[int(properties["NE_ID"])] = (str(properties["ADM0_A3"]), str(properties["NAME"]))
    return found


def read_raster(path: Path, box: Box = EUROPE) -> NDArray[np.uint32]:
    """The finer raster of `NE_ID`s, rows from the south; it must cover the box exactly."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages.first
        scale = page.tags["ModelPixelScaleTag"].value
        tiepoint = page.tags["ModelTiepointTag"].value
        values = np.asarray(page.asarray(), dtype=np.uint32)
    step = box.step / FINER
    if values.shape != (box.rows * FINER, box.columns * FINER):
        raise ValueError(f"{path.name} is {values.shape}, not {FINER} times the box's cells")
    if not np.allclose(scale[:2], (step, step), atol=1e-9, rtol=0):
        raise ValueError(f"{path.name}'s pixels are {scale[:2]}, not {step}")
    north = box.south - box.step / 2 + box.rows * box.step
    if not np.allclose((tiepoint[3], tiepoint[4]), (box.west - box.step / 2, north), atol=1e-6):
        raise ValueError(f"{path.name} does not start at the box's north-west corner")
    return np.flipud(values)


def majority(fine: NDArray[np.integer], finer: int = FINER) -> NDArray[np.uint16]:
    """Each cell's most common nonzero code among its finer cells, 0 if all are zero; a tie
    goes to the smaller code. `fine` must hold codes below 65,536."""
    rows, columns = fine.shape[0] // finer, fine.shape[1] // finer
    codes = int(fine.max()) + 1
    result = np.zeros((rows, columns), dtype=np.uint16)
    for row in range(rows):
        band = fine[row * finer : (row + 1) * finer].astype(np.int64)
        blocks = band.reshape(finer, columns, finer).transpose(1, 0, 2).reshape(columns, -1)
        flat = (np.arange(columns)[:, np.newaxis] * codes + blocks).reshape(-1)
        counts = np.bincount(flat, minlength=columns * codes).reshape(columns, codes)
        counts[:, 0] = 0
        best = counts.argmax(axis=1)
        result[row] = np.where(counts[np.arange(columns), best] > 0, best, 0)
    return result


def build(raster: Path, geojson: Path, out: Path, box: Box = EUROPE) -> Path:
    """Writes `countries.npz` from the finer raster of `NE_ID`s and the countries' file."""
    countries = read_countries(geojson)
    fine = read_raster(raster, box)
    ids = np.unique(fine)
    present = [int(ne_id) for ne_id in ids if ne_id != 0]
    unknown = [ne_id for ne_id in present if ne_id not in countries]
    if unknown:
        raise ValueError(f"the raster holds NE_IDs the countries' file has not: {unknown[:5]}")
    if len(present) >= np.iinfo(np.uint16).max:
        raise ValueError(f"{len(present)} countries do not fit a uint16")
    # Codes from 1 in the order of the countries' ADM0_A3, 0 for the sea.
    ordered = sorted(present, key=lambda ne_id: countries[ne_id][0])
    lookup = np.zeros(int(ids.max()) + 1, dtype=np.uint32)
    lookup[ordered] = np.arange(1, len(ordered) + 1)
    code = majority(lookup[fine])
    out.mkdir(parents=True, exist_ok=True)
    path = out / ASSET
    np.savez_compressed(
        path,
        code=code,
        iso=np.array([SEA[0], *(countries[ne_id][0] for ne_id in ordered)]),
        name=np.array([SEA[1], *(countries[ne_id][1] for ne_id in ordered)]),
        south=np.float64(box.south),
        west=np.float64(box.west),
        step=np.float64(box.step),
        source=np.array(SOURCE),
    )
    return path


def describe(path: Path) -> list[str]:
    """The largest countries by cells, for the build's summary."""
    with np.load(path) as held:
        code, iso, name = held["code"], held["iso"], held["name"]
    counts = np.bincount(code.reshape(-1), minlength=len(iso))
    order = np.argsort(counts)[::-1]
    lines = [f"{len(iso) - 1} countries; sea {counts[0]} cells of {code.size}."]
    lines += [f"{iso[i]} {name[i]}: {counts[i]} cells" for i in order if i != 0][:10]
    return lines
