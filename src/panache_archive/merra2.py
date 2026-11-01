"""MERRA-2's daily aerosol surface concentrations over the whole domain, from NASA's Cloud OPeNDAP
(Panache's D72).

Panache's second opinion on an episode's aerosols (its D68) reads three of MERRA-2's species at
the place, every day since 2013. Giovanni gives them one place at a time, in 10 to 15 s; this
archive holds every cell of the CAMS domain at once, read one cell at a time from afar.

Data: Global Modeling and Assimilation Office (GMAO, 2015), MERRA-2 tavg1_2d_aer_Nx (M2T1NXAER)
V5.12.4, doi:10.5067/KLICLTZ8EM9D: hourly surface mass concentrations on a 0.5° by 0.625° grid,
served by NASA's GES DISC. Cloud OPeNDAP cuts each daily file to the domain and the three
species (~1.8 MB a day, 6 to 9 s a request; 8 at once, a year in ~7 min), with an Earthdata
Login token (`EARTHDATA_TOKEN`). A few months were reprocessed: their files are `MERRA2_401`,
where the others are `MERRA2_400`.

- **Days**: UTC daily means of at least 18 valid hours, in μg/m³, as Panache makes them from
  Giovanni's hours.
- **Grid**: centres every 0.5° from 30° N and every 0.625° from 25° W: 85 by 113 cells.
- **Assets**: one per year, `merra2-2021.cells` (version 2 of the format, for its two steps, one
  cell a frame); the year under way as its months out so far, `merra2-2026-01-09.cells`,
  replaced as MERRA-2 comes out, a few weeks after each month.
"""

import concurrent.futures
import csv
import io
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import netCDF4
import numpy as np

from panache_archive import cells
from panache_archive.build import ALL_MONTHS, check_months, read_metadata, span
from panache_archive.check import (
    EUROPE_PLACES,
    Comparison,
    Place,
    compare_days,
)
from panache_archive.daily import MIN_HOURS_PER_DAY
from panache_archive.grid import FloatArray, nearest

COLLECTION: Final = "C1276812830-GES_DISC"
SHORT_NAME: Final = "M2T1NXAER.5.12.4"
OPENDAP_URL: Final = f"https://opendap.earthdata.nasa.gov/collections/{COLLECTION}/granules/"
GIOVANNI_URL: Final = "https://api.giovanni.earthdata.nasa.gov/timeseries"
GIOVANNI_COLLECTION: Final = "M2T1NXAER_5_12_4"
STREAMS: Final = ("400", "401")
"""File streams: 401 holds the months GMAO reprocessed (September 2020, summer 2021)."""
SPECIES: Final = ("DUSMASS", "OCSMASS", "BCSMASS")
"""Dust (five bins, up to 20 µm), organic carbon and black carbon, the statistics' order."""
UNITS: Final = "kg m-3"
UG_PER_KG: Final = 1e9
FIRST_YEAR: Final = 2013
LAT_STEP: Final = 0.5
LON_STEP: Final = 0.625
ROWS: Final = slice(240, 325)
"""30° N to 72° N on MERRA-2's grid, which starts at 90° S."""
COLS: Final = slice(248, 361)
"""25° W to 45° E on MERRA-2's grid, which starts at 180° W."""
LATS: Final[FloatArray] = np.round(-90 + LAT_STEP * np.arange(ROWS.start, ROWS.stop), 4)
LONS: Final[FloatArray] = np.round(-180 + LON_STEP * np.arange(COLS.start, COLS.stop), 4)
HOURS: Final = 24
CONCURRENCY: Final = 8
CELLS_PER_FRAME: Final = 1
"""One cell a frame: three species of a year, ~2 KB a read."""
DOI: Final = "10.5067/KLICLTZ8EM9D"
LICENCE: Final = "NASA open data, no restrictions: cite the DOI"
ATTRIBUTION: Final = (
    "Global Modeling and Assimilation Office (GMAO) (2015), MERRA-2 tavg1_2d_aer_Nx: 2d,1-Hourly,"
    "Time-averaged,Single-Level,Assimilation,Aerosol Diagnostics V5.12.4, Greenbelt, MD, USA, "
    "Goddard Earth Sciences Data and Information Services Center (GES DISC)."
)
TOLERANCE: Final = 0.005 + 1e-6
"""Against Giovanni's hours, which are the same values: the archive's rounding to 0.01 μg/m³."""

logger = logging.getLogger(__name__)
Fetch = Callable[[str], bytes]
"""Fetches a URL with the token and returns the body; raises `urllib.error.HTTPError`."""


@dataclass(frozen=True, slots=True)
class Merra2Job:
    """One asset: the domain's daily species over consecutive months of a year."""

    year: int
    months: tuple[int, ...] = ALL_MONTHS

    def __post_init__(self) -> None:
        check_months(self.months)

    @property
    def asset_name(self) -> str:
        """Such as `merra2-2021.cells`, or `merra2-2026-01-09.cells` for a year under way."""
        return f"merra2-{self.year}{span(self.months)}.cells"

    @property
    def first_day(self) -> date:
        return date(self.year, self.months[0], 1)

    @property
    def days(self) -> int:
        last = self.months[-1]
        return (date(self.year + last // 12, last % 12 + 1, 1) - self.first_day).days


def token_fetch(token: str) -> Fetch:
    """Reads a URL with the Earthdata token as a bearer, retried once on a server error."""

    def fetch(url: str) -> bytes:
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        for attempt in (1, 2):
            try:
                with urllib.request.urlopen(request, timeout=180) as answer:
                    return bytes(answer.read())
            except urllib.error.HTTPError as error:
                if error.code < 500 or attempt == 2:
                    raise
            time.sleep(5)
        raise AssertionError("unreachable: the last attempt returns or raises")

    return fetch


def day_url(stream: str, day: date) -> str:
    """The domain's three species on `day`, cut by OPeNDAP (DAP2's netCDF-4 answer)."""
    granule = f"{SHORT_NAME}:MERRA2_{stream}.tavg1_2d_aer_Nx.{day:%Y%m%d}.nc4"
    box = f"[0:1:{HOURS - 1}][{ROWS.start}:1:{ROWS.stop - 1}][{COLS.start}:1:{COLS.stop - 1}]"
    constraint = ",".join(f"{species}{box}" for species in SPECIES)
    constraint += f",lat[{ROWS.start}:1:{ROWS.stop - 1}],lon[{COLS.start}:1:{COLS.stop - 1}]"
    return (
        f"{OPENDAP_URL}{urllib.parse.quote(granule)}.nc4?{urllib.parse.quote(constraint, safe=',')}"
    )


def read_day(day: date, fetch: Fetch) -> FloatArray:
    """The day's means, `(species, lat, lon)` in μg/m³, NaN with fewer than 18 hours."""
    return day_means(day, fetch_day(day, fetch))


def fetch_day(day: date, fetch: Fetch) -> bytes:
    """The domain's cut of `day`, from its `MERRA2_400` file, else its reprocessed one."""
    for stream in STREAMS[:-1]:
        try:
            return fetch(day_url(stream, day))
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
    return fetch(day_url(STREAMS[-1], day))


def day_means(day: date, data: bytes) -> FloatArray:
    """The means of a day's cut, `(species, lat, lon)`. netCDF4 (HDF5) is not thread-safe: call
    it from one thread at a time."""
    with netCDF4.Dataset(f"merra2-{day:%Y%m%d}.nc4", memory=data) as dataset:
        lats = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        lons = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
        same_shape = lats.shape == LATS.shape and lons.shape == LONS.shape
        if not (same_shape and np.allclose(lats, LATS) and np.allclose(lons, LONS)):
            raise ValueError(f"{day}: the cut is not the domain's grid")
        means = []
        for species in SPECIES:
            variable = dataset.variables[species]
            if variable.units != UNITS or variable.shape != (HOURS, LATS.size, LONS.size):
                raise ValueError(f"{day}: {species} is {variable.shape} in {variable.units}")
            hours = np.ma.filled(np.ma.asarray(variable[:], dtype=np.float64), np.nan)
            valid = np.isfinite(hours)
            count = valid.sum(axis=0)
            total = np.where(valid, hours, 0.0).sum(axis=0)
            mean = total / np.maximum(count, 1) * UG_PER_KG
            means.append(np.where(count >= MIN_HOURS_PER_DAY, mean, np.nan))
    return np.stack(means)


def build(job: Merra2Job, out_dir: Path, fetch: Fetch, *, concurrency: int = CONCURRENCY) -> Path:
    """Build `job`'s asset in `out_dir`; returns its path. The days are fetched `concurrency` at
    once, and read one by one as they come, since netCDF4 is not thread-safe."""
    days = [job.first_day + timedelta(days=n) for n in range(job.days)]
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        fetched = pool.map(lambda day: fetch_day(day, fetch), days)
        daily = np.stack([day_means(day, data) for day, data in zip(days, fetched, strict=True)])
    logger.info("%s: %d days read in %.0f s", job.asset_name, len(days), time.monotonic() - started)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / job.asset_name
    cells.write(
        path,
        LATS,
        LONS,
        job.first_day,
        [daily[:, k] for k in range(len(SPECIES))],
        metadata=_metadata(),
        cells_per_frame=CELLS_PER_FRAME,
    )
    logger.info("%s: %.1f MB", path.name, path.stat().st_size / 1e6)
    return path


def last_month_out(year: int, today: date, fetch: Fetch) -> int:
    """The last month of `year` whose last day MERRA-2 holds, 0 if none: it comes out a few
    weeks after each month."""
    for month in range(12 if year < today.year else today.month - 1, 0, -1):
        last = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
        if _exists(last, fetch):
            return month
    return 0


def plan(
    years: Sequence[int], built: set[str], today: date, fetch: Fetch
) -> tuple[list[Merra2Job], list[str]]:
    """The jobs the release lacks (whole years, and the year under way as its months out), and
    the assets they supersede (an earlier part of a year)."""
    jobs: list[Merra2Job] = []
    superseded: list[str] = []
    for year in years:
        months = 12 if year < today.year - 1 else last_month_out(year, today, fetch)
        if months == 0:
            continue
        job = Merra2Job(year, tuple(range(1, months + 1)))
        older = sorted(n for n in built if n.startswith(f"merra2-{year}") and n != job.asset_name)
        if job.asset_name not in built:
            jobs.append(job)
        superseded += older
    return jobs, superseded


def compare(
    asset: Path, places: Sequence[Place] = EUROPE_PLACES, fetch: Fetch | None = None
) -> list[Comparison]:
    """The asset against Giovanni's hours at each place, species by species."""
    if read_metadata(asset).get("source") != "merra2":
        raise ValueError(f"{asset.name} is not a MERRA-2 asset")
    if fetch is None:
        raise ValueError("comparing with Giovanni needs the Earthdata token")
    header = cells.read_header(asset)
    last = header.first_day + timedelta(days=header.days - 1)
    comparisons: list[Comparison] = []
    for place in places:
        row, column = nearest(header.lats, place.lat), nearest(header.lons, place.lon)
        cell = (float(header.lats[row]), float(header.lons[column]))
        values = cells.to_values(cells.read_cell(asset, row, column), header.scale)
        for k, species in enumerate(SPECIES):
            reference = giovanni_daily(species, place, header.first_day, last, fetch)
            comparisons.append(compare_days(place, cell, species, values[k], reference, TOLERANCE))
    return comparisons


def giovanni_daily(species: str, place: Place, first: date, last: date, fetch: Fetch) -> FloatArray:
    """Giovanni's daily means of `species` at the cell nearest `place`, in μg/m³."""
    query = urllib.parse.urlencode(
        {
            "data": f"{GIOVANNI_COLLECTION}_{species}",
            "location": f"[{place.lat},{place.lon}]",
            "time": f"{first.isoformat()}T00:00:00/{last.isoformat()}T23:59:59",
        }
    )
    text = fetch(f"{GIOVANNI_URL}?{query}").decode()
    head, _, table = text.partition("Timestamp (UTC),Data")
    header = dict(line.split(",", 1) for line in head.splitlines() if "," in line)
    undefined = float(header["undef"])
    days = (last - first).days + 1
    sums, counts = np.zeros(days), np.zeros(days)
    for stamp, value in csv.reader(io.StringIO(table.strip())):
        index = (datetime.fromisoformat(stamp).replace(tzinfo=UTC).date() - first).days
        number = float(value)
        if 0 <= index < days and number != undefined and np.isfinite(number):
            sums[index] += number
            counts[index] += 1
    return np.where(counts >= MIN_HOURS_PER_DAY, sums / np.maximum(counts, 1) * UG_PER_KG, np.nan)


def _exists(day: date, fetch: Fetch) -> bool:
    """Whether MERRA-2 holds `day`, from its OPeNDAP metadata (a few kilobytes)."""
    for stream in STREAMS:
        granule = f"{SHORT_NAME}:MERRA2_{stream}.tavg1_2d_aer_Nx.{day:%Y%m%d}.nc4"
        try:
            fetch(f"{OPENDAP_URL}{urllib.parse.quote(granule)}.dds")
            return True
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
    return False


def _metadata() -> dict[str, Any]:
    return {
        "source": "merra2",
        "statistics": list(SPECIES),
        "unit": "μg/m³",
        "days": "UTC",
        "grid": "0.5° by 0.625°",
        "collection": SHORT_NAME,
        "doi": DOI,
        "licence": LICENCE,
        "attribution": ATTRIBUTION,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
