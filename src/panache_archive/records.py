"""Each cell's record in one frame (Panache's D134): a pollutant's validated years over one CAMS
grid, stacked into one `.cells` file of one cell a frame, from `cams-europe-v1`'s yearly assets.

A click on a place reads a pollutant's year in two small requests (`cells`): ~230 requests for the
ten pollutants since 2013, 8 cells a frame. A record gives the same cell's validated years at once,
one cell a frame: two requests a grid, as the grid moved half a cell in 2020 (Panache's D52), and
none of its seven neighbours. Alone, a cell compresses less: wildfire PM10 2020-2024 takes 190 MB
where its five years take 110 MB, but a cell's five years read 0.65 KB where they read 2.9 KB. The
values are the yearly assets' very hundredths (`check`).

A record holds consecutive validated years of one grid: 2013-2019 on the grid centred on x.x0°
(its first five years stop at 70° N: their northern rows are null), 2020 onwards on today's. An
interim year stays read a year at a time; a longer record is built once the archive gains a
validated year, and the reader takes the longest.
"""

import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import numpy as np

from panache_archive import cells

RELEASE: Final = "cams-europe-records-v1"
SOURCE_RELEASE: Final = "cams-europe-v1"
GRID_CHANGE_YEAR: Final = 2020
"""The first year of CAMS's grid centred on x.x5° (Panache's D52): a record never spans it."""
VALIDATED: Final = re.compile(r"cams-(?P<variable>[a-z0-9_]+)-(?P<year>\d{4})-validated\.cells")
RECORD: Final = re.compile(r"cams-(?P<variable>[a-z0-9_]+)-(?P<first>\d{4})-(?P<last>\d{4})\.cells")
BAND_ROWS: Final = 10
"""Rows read at once from every year: ~150 MB of hundredths for ozone's two statistics."""
CHECKED_CELLS: Final = 400
"""Cells of a record compared, every day, with their yearly assets, beside the grid's corners."""


@dataclass(frozen=True, slots=True)
class Record:
    """A pollutant's consecutive validated years over one grid."""

    variable: str
    first: int
    last: int

    @property
    def name(self) -> str:
        return f"cams-{self.variable}-{self.first}-{self.last}.cells"

    @property
    def years(self) -> range:
        return range(self.first, self.last + 1)

    @property
    def sources(self) -> list[str]:
        """The yearly assets it is stacked from, in order."""
        return [f"cams-{self.variable}-{year}-validated.cells" for year in self.years]


def planned(assets: Iterable[str], built: Iterable[str]) -> list[Record]:
    """The records the validated assets among `assets` make that `built` lacks: for each
    pollutant and grid, its longest run of consecutive years ending on the grid's last one, if it
    holds two years or more (one year is as fast read from its own asset)."""
    years: dict[str, set[int]] = {}
    for name in assets:
        found = VALIDATED.fullmatch(name)
        if found is not None:
            years.setdefault(found["variable"], set()).add(int(found["year"]))
    done = set(built)
    wanted: list[Record] = []
    for variable, held in sorted(years.items()):
        for era in _eras(held):
            record = Record(variable, era[0], era[-1])
            if len(era) >= 2 and record.name not in done:
                wanted.append(record)
    return wanted


def matrix(records: Sequence[Record]) -> str:
    """The build jobs, as GitHub Actions' matrix reads them."""
    return json.dumps(
        [
            {
                "record": r.name,
                "variable": r.variable,
                "first": r.first,
                "last": r.last,
                "sources": " ".join(r.sources),
            }
            for r in records
        ]
    )


def build(record: Record, folder: Path, out: Path, *, band_rows: int = BAND_ROWS) -> Path:
    """The record from its yearly assets in `folder`, written to `out`, a band of rows at a time.

    The years must share the grid's south-west corner, step, columns and statistics, and each
    must hold its whole calendar year; a year with fewer rows (before 2018, up to 70° N) leaves
    the northern ones null.
    """
    paths = [folder / name for name in record.sources]
    headers = [cells.read_header(path) for path in paths]
    _check_years(record, headers)
    grid = max(headers, key=lambda h: h.rows)
    header = cells.Header(
        statistics=grid.statistics,
        cells_per_frame=1,
        first_lat=grid.first_lat,
        first_lon=grid.first_lon,
        step=grid.step,
        rows=grid.rows,
        columns=grid.columns,
        first_day=date(record.first, 1, 1),
        days=sum(h.days for h in headers),
        lon_step=grid.lon_step,
    )
    metadata = {
        "variable": record.variable,
        "years": list(record.years),
        "sources": record.sources,
        "release": SOURCE_RELEASE,
        "licence": "CC-BY-4.0",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    blocks = _blocks(header, paths, headers, band_rows)
    cells.write_frames(out, header, blocks, metadata=metadata)
    return out


def check(path: Path, folder: Path, *, cells_checked: int = CHECKED_CELLS, seed: int = 0) -> int:
    """How many cells of the record at `path` were compared, every day, with the yearly assets in
    `folder` it names. Raises `ValueError` at the first that differs."""
    header = cells.read_header(path)
    sources = [folder / name for name in cells.read_metadata(path)["sources"]]
    rng = np.random.default_rng(seed)
    corners = [(0, 0), (0, header.columns - 1), (header.rows - 1, 0)]
    corners.append((header.rows - 1, header.columns - 1))
    drawn = zip(
        rng.integers(0, header.rows, cells_checked).tolist(),
        rng.integers(0, header.columns, cells_checked).tolist(),
        strict=True,
    )
    compared = 0
    for row, column in [*corners, *drawn]:
        record = cells.read_cell(path, row, column)
        day = 0
        for source in sources:
            year = cells.read_header(source)
            expected = (
                cells.read_cell(source, row, column)
                if row < year.rows
                else np.full((year.statistics, year.days), cells.NULL, dtype=np.int32)
            )
            if not np.array_equal(record[:, day : day + year.days], expected):
                raise ValueError(f"{path.name} differs from {source.name} at ({row}, {column})")
            day += year.days
        compared += 1
    return compared


def _eras(years: set[int]) -> Iterator[list[int]]:
    """Per grid, the run of consecutive years ending on the grid's last year held."""
    for era in (
        [y for y in years if y < GRID_CHANGE_YEAR],
        [y for y in years if y >= GRID_CHANGE_YEAR],
    ):
        if not era:
            continue
        run = [max(era)]
        while run[0] - 1 in era:
            run.insert(0, run[0] - 1)
        yield run


def _check_years(record: Record, headers: Sequence[cells.Header]) -> None:
    grids = {
        (h.first_lat, h.first_lon, h.step, h.lon_step, h.columns, h.statistics) for h in headers
    }
    if len(grids) != 1:
        raise ValueError(f"{record.name}: its years lie on other grids: {sorted(grids)}")
    for year, header in zip(record.years, headers, strict=True):
        days = (date(year + 1, 1, 1) - date(year, 1, 1)).days
        if (header.first_day, header.days) != (date(year, 1, 1), days):
            raise ValueError(
                f"{record.name}: {year} holds {header.days} days from {header.first_day}"
            )


def _blocks(
    header: cells.Header,
    paths: Sequence[Path],
    headers: Sequence[cells.Header],
    band_rows: int,
) -> Iterator[cells.IntArray]:
    """Each cell's frame, `(statistic, 1, day)`, row by row: every year's band of rows at once."""
    for start in range(0, header.rows, band_rows):
        rows = range(start, min(start + band_rows, header.rows))
        band = np.full(
            (header.statistics, header.days, len(rows), header.columns), cells.NULL, np.int32
        )
        day = 0
        for path, year in zip(paths, headers, strict=True):
            inside = range(rows.start, min(rows.stop, year.rows))
            if inside:
                band[:, day : day + year.days, : len(inside)] = cells.read_region(
                    path, inside, range(year.columns)
                )
            day += year.days
        # (row, column, statistic, day): each cell's frame is then one contiguous slice.
        by_cell = np.ascontiguousarray(band.transpose(2, 3, 0, 1))
        for row in range(len(rows)):
            for column in range(header.columns):
                yield by_cell[row, column][:, np.newaxis, :]
