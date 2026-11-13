"""EDGAR's primary PM2.5 by sector for Panache's Lagrangian footprints (Panache's D86).

CAMS-GLOB-ANT (`emissions`) has black and organic carbon but no particulate mass. EDGAR v8.1
(Crippa et al. 2024, European Commission, Joint Research Centre,
https://edgar.jrc.ec.europa.eu/dataset_ap81) has PM2.5 itself: one flux map a year and sector,
kg m⁻² s⁻¹, at 0.1°, 1970-2022, on the same cells as CAMS-GLOB-ANT.

- **Use conditions** (read on 2026-11-13 in the JRC Data Catalogue and on EDGAR's page): the
  European Commission's reuse notice, reuse authorised provided the source is acknowledged;
  anonymous access. EDGAR asks for a reference to its v8.1 air pollutant website.
- **Source files**: one zip per sector, `<base>/PM2.5/<SECTOR>/<SECTOR>_flx_nc.zip` (3.2 GB for
  the 24), each holding `v8.1_FT2022_AP_PM2.5_<year>_<SECTOR>_flx.nc`, a `fluxes` variable on
  (lat, lon).
- **Kept** (`SURFACE`): the 20 sectors emitted at or near the ground, each under a name of at
  most 8 letters. Aviation's climb, cruise and supersonic sectors (`ALOFT`) are emitted aloft:
  a surface footprint does not see them. They are read only to check that every sector adds up
  to EDGAR's own `TOTALS`. A sector with no file for a year emitted nothing that year.
- **Assets**: `emissions-edgar_pm25-<year>.emissions`, the format of `emissions` with one
  period a year (EDGAR's maps are yearly) and a scale of 1e10 (EDGAR's largest cells are ten
  times CAMS-GLOB-ANT's).
"""

import logging
import shutil
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import netCDF4
import numpy as np
from numpy.typing import NDArray

from panache_archive.emissions import (
    EUROPE,
    SECONDS_PER_YEAR,
    Box,
    Header,
    _areas,
    _positions,
    write,
)

Download = Callable[[str, Path], None]
"""Fetches a URL into a file."""

logger = logging.getLogger(__name__)

BASE: Final = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/v81_FT2022_AP_new"
SUBSTANCE: Final = "PM2.5"
SURFACE: Final = {
    "AGS": "AGS",
    "AWB": "AWB",
    "CHE": "CHE",
    "ENE": "ENE",
    "FOO_PAP": "FOO_PAP",
    "IND": "IND",
    "IRO": "IRO",
    "MNM": "MNM",
    "NFE": "NFE",
    "NMM": "NMM",
    "PRO_FFF": "PRO_FFF",
    "PRU_SOL": "PRU_SOL",
    "RCO": "RCO",
    "REF_TRF": "REF_TRF",
    "SWD_INC": "SWD_INC",
    "SWD_LDF": "SWD_LDF",
    "TNR_Aviation_LTO": "AVI_LTO",
    "TNR_Other": "TNR_OTH",
    "TNR_Ship": "TNR_SHIP",
    "TRO": "TRO",
}
"""EDGAR's folder of each sector kept, and its name in the asset."""
ALOFT: Final = ("TNR_Aviation_CDS", "TNR_Aviation_CRS", "TNR_Aviation_SPS")
TOTAL: Final = "TOTALS"
SCALE: Final = 1e10
FIRST_YEAR: Final = 2013
LAST_YEAR: Final = 2022
ATTEMPTS: Final = 3


def fetch(url: str, target: Path) -> None:
    """Downloads a file without holding it in memory, trying again when it is cut short."""
    for attempt in range(ATTEMPTS):
        request = urllib.request.Request(url, headers={"User-Agent": "panache-archive"})
        try:
            with (
                urllib.request.urlopen(request, timeout=300) as response,
                target.open("wb") as file,
            ):
                shutil.copyfileobj(response, file, length=1 << 20)
            return
        except OSError:
            if attempt == ATTEMPTS - 1:
                raise
            logger.warning("%s: cut short, trying again", url)


def fluxes(archive: zipfile.ZipFile, year: int, box: Box) -> NDArray[np.float32] | None:
    """A sector's year over the box, kg m⁻² s⁻¹, shaped (row, column); nothing where EDGAR has
    no value. None when the sector has no file for the year: EDGAR writes none for a year a
    sector emits nothing (supersonic flight after 2003)."""
    names = [name for name in archive.namelist() if f"_{year}_" in name and name.endswith(".nc")]
    if not names:
        return None
    if len(names) > 1:
        raise ValueError(f"expected one file for {year}, found {len(names)}")
    with netCDF4.Dataset(names[0], memory=archive.read(names[0])) as dataset:
        rows = _positions(np.asarray(dataset.variables["lat"][:], dtype=np.float64), box.lats)
        columns = _positions(np.asarray(dataset.variables["lon"][:], dtype=np.float64), box.lons)
        cut = np.ma.filled(dataset.variables["fluxes"][rows, columns], 0.0)
    return np.nan_to_num(cut, nan=0.0).astype(np.float32)


def build(
    years: Sequence[int],
    out_dir: Path,
    workdir: Path,
    download: Download = fetch,
    *,
    box: Box = EUROPE,
) -> list[Path]:
    """Builds `emissions-edgar_pm25-<year>.emissions` for each year, a sector's zip at a time."""
    workdir.mkdir(parents=True, exist_ok=True)
    kept: dict[int, list[NDArray[np.float32]]] = {year: [] for year in years}
    summed = dict.fromkeys(years, 0.0)
    totals = dict.fromkeys(years, 0.0)
    for folder in (*SURFACE, *ALOFT, TOTAL):
        source = workdir / f"{folder}.zip"
        download(f"{BASE}/{SUBSTANCE}/{folder}/{folder}_flx_nc.zip", source)
        try:
            with zipfile.ZipFile(source) as archive:
                for year in years:
                    grid = fluxes(archive, year, box)
                    if grid is None:
                        if folder == TOTAL:
                            raise ValueError(f"EDGAR's total has no file for {year}")
                        # The check against the total below catches a file that should exist.
                        logger.warning("%s has no file for %d: taken as zero", folder, year)
                        grid = np.zeros((box.rows, box.columns), dtype=np.float32)
                    if folder == TOTAL:
                        totals[year] = float(grid.sum(dtype=np.float64))
                        continue
                    summed[year] += float(grid.sum(dtype=np.float64))
                    if folder in SURFACE:
                        kept[year].append(grid)
        finally:
            source.unlink()
        logger.info("%s read", folder)

    out_dir.mkdir(parents=True, exist_ok=True)
    areas = _areas(box.lats, box.step)[:, np.newaxis]
    paths = []
    for year in years:
        # No absolute tolerance: these are sums of kg m⁻² s⁻¹, far below numpy's default.
        if not np.isclose(summed[year], totals[year], rtol=0.01, atol=0.0):
            raise ValueError(
                f"{year}: the sectors add up to {summed[year]:.4g}, EDGAR's total to "
                f"{totals[year]:.4g}"
            )
        values = np.stack(kept[year])[np.newaxis]
        path = out_dir / f"emissions-edgar_pm25-{year}.emissions"
        write(path, Header(year, tuple(SURFACE.values()), box, periods=1, scale=SCALE), values)
        yearly = float((values[0].sum(axis=0, dtype=np.float64) * areas).sum())
        logger.info(
            "%s: %.0f kt a year at the surface over the box, %.1f MB",
            path.name,
            yearly * SECONDS_PER_YEAR / 1e6,
            path.stat().st_size / 1e6,
        )
        paths.append(path)
    return paths
