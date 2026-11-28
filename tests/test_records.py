import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from panache_archive import cells, records
from panache_archive.cli import main
from panache_archive.grid import FloatArray
from panache_archive.records import Record

LATS = np.round(30.0 + 0.1 * np.arange(3), 2)
LONS = np.round(-25.0 + 0.1 * np.arange(11), 2)
"""Two frames a row in the yearly assets (8 and 3 cells), eleven in a record."""
OZONE = Record("ozone", 2016, 2018)


def _year(folder: Path, variable: str, year: int, *, rows: int = 3, statistics: int = 2) -> Path:
    """A yearly asset of `rows` rows, every value telling its year, day, statistic and cell."""
    days = (date(year + 1, 1, 1) - date(year, 1, 1)).days
    day = np.arange(days)[:, None, None]
    cell = np.arange(rows * LONS.size).reshape(rows, LONS.size)[None, :, :]
    columns: list[FloatArray] = [
        (year - 2000) * 1000.0 + day + cell / 100 + statistic * 0.5
        for statistic in range(statistics)
    ]
    columns[0][3, 0, 2] = np.nan
    path = folder / f"cams-{variable}-{year}-validated.cells"
    cells.write(path, LATS[:rows], LONS, date(year, 1, 1), columns, metadata={"year": year})
    return path


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    # 2016 stops a row south of the others, as the reanalysis stopped at 70° N until 2017.
    _year(tmp_path, "ozone", 2016, rows=2)
    _year(tmp_path, "ozone", 2017)
    _year(tmp_path, "ozone", 2018)
    return tmp_path


# planned


def test_plans_a_record_per_pollutant_and_grid() -> None:
    assets = [
        *(f"cams-pm10-{year}-validated.cells" for year in range(2013, 2025)),
        "cams-pm10-2025-interim.cells",
        *(f"cams-dust-{year}-validated.cells" for year in range(2016, 2025)),
        "cams-pm10-2025-interim.parquet",
    ]

    planned = records.planned(assets, built=[])

    assert [r.name for r in planned] == [
        "cams-dust-2016-2019.cells",
        "cams-dust-2020-2024.cells",
        "cams-pm10-2013-2019.cells",
        "cams-pm10-2020-2024.cells",
    ]


def test_plans_only_what_is_not_built_and_holds_two_years() -> None:
    assets = [f"cams-pm10-{year}-validated.cells" for year in (2013, 2018, 2019, 2020)]

    planned = records.planned(assets, built=["cams-pm10-2018-2019.cells"])

    assert planned == []


def test_a_record_runs_back_from_its_grid_s_last_year() -> None:
    assets = [f"cams-no2-{year}-validated.cells" for year in (2020, 2022, 2023, 2024)]

    assert records.planned(assets, built=[]) == [Record("no2", 2022, 2024)]


def test_the_matrix_names_each_record_and_its_sources() -> None:
    [job] = json.loads(records.matrix([OZONE]))

    assert job == {
        "record": "cams-ozone-2016-2018.cells",
        "variable": "ozone",
        "first": 2016,
        "last": 2018,
        "sources": (
            "cams-ozone-2016-validated.cells cams-ozone-2017-validated.cells "
            "cams-ozone-2018-validated.cells"
        ),
    }


# build


def test_a_record_holds_every_year_of_each_cell_in_one_frame(folder: Path) -> None:
    path = records.build(OZONE, folder, folder / "out" / OZONE.name, band_rows=2)

    header = cells.read_header(path)
    assert header.cells_per_frame == 1
    assert (header.first_day, header.days, header.statistics) == (date(2016, 1, 1), 1096, 2)
    assert (header.rows, header.columns) == (3, 11)
    np.testing.assert_array_equal(header.lats, LATS)
    for row, column in [(0, 0), (1, 7), (1, 8), (2, 10)]:
        cell = cells.read_cell(path, row, column)
        day = 0
        for year in OZONE.years:
            source = folder / f"cams-ozone-{year}-validated.cells"
            year_days = cells.read_header(source).days
            if row < cells.read_header(source).rows:
                expected = cells.read_cell(source, row, column)
                np.testing.assert_array_equal(cell[:, day : day + year_days], expected)
            day += year_days


def test_a_year_without_the_northern_rows_leaves_them_null(folder: Path) -> None:
    path = records.build(OZONE, folder, folder / OZONE.name)

    north = cells.read_cell(path, 2, 4)

    assert (north[:, :366] == cells.NULL).all()  # 2016, a leap year
    assert (north[:, 366:] != cells.NULL).all()


def test_a_missing_value_stays_missing(folder: Path) -> None:
    path = records.build(OZONE, folder, folder / OZONE.name)

    assert cells.read_cell(path, 0, 2)[0, 3] == cells.NULL


def test_a_record_says_what_it_is_stacked_from(folder: Path) -> None:
    path = records.build(OZONE, folder, folder / OZONE.name)

    metadata = cells.read_metadata(path)
    assert metadata["years"] == [2016, 2017, 2018]
    assert metadata["sources"] == OZONE.sources
    assert metadata["release"] == "cams-europe-v1"


def test_years_on_another_grid_are_refused(folder: Path) -> None:
    _year(folder, "pm10", 2016, statistics=1)
    _year(folder, "pm10", 2017, statistics=2)

    with pytest.raises(ValueError, match="other grids"):
        records.build(Record("pm10", 2016, 2017), folder, folder / "x.cells")


def test_a_year_that_is_not_whole_is_refused(folder: Path) -> None:
    cells.write(
        folder / "cams-pm10-2017-validated.cells",
        LATS,
        LONS,
        date(2017, 2, 1),
        [np.zeros((10, LATS.size, LONS.size))],
        metadata={},
    )
    _year(folder, "pm10", 2016, statistics=1)

    with pytest.raises(ValueError, match="2017 holds 10 days"):
        records.build(Record("pm10", 2016, 2017), folder, folder / "x.cells")


# check


def test_checks_a_record_against_its_yearly_assets(folder: Path) -> None:
    path = records.build(OZONE, folder, folder / "out" / OZONE.name)

    assert records.check(path, folder, cells_checked=20) == 24


def test_a_record_that_differs_from_its_years_is_refused(folder: Path) -> None:
    path = records.build(OZONE, folder, folder / "out" / OZONE.name)
    changed = folder / "cams-ozone-2017-validated.cells"
    header, values = cells.read_all(changed)
    values[0, 100, 0, 0] += 1  # one value of a corner, a hundredth off
    cells.write(
        changed,
        header.lats,
        header.lons,
        header.first_day,
        [cells.to_values(values[s]) for s in range(header.statistics)],
        metadata={},
    )

    with pytest.raises(ValueError, match=r"differs from cams-ozone-2017-validated.cells"):
        records.check(path, folder, cells_checked=0)


# command line


def test_the_commands_plan_build_and_check(
    folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    listing = tmp_path / "assets.txt"
    listing.write_text("\n".join(OZONE.sources) + "\ncams-pm10-2019-validated.cells\n")

    assert main(["records-plan", "--assets", str(listing)]) == 0
    [job] = json.loads(capsys.readouterr().out)
    assert job["record"] == OZONE.name

    out = tmp_path / "dist"
    args = ["--variable", "ozone", "--first", "2016", "--last", "2018"]
    assert main(["records-build", *args, "--folder", str(folder), "--out", str(out)]) == 0
    summary = tmp_path / "summary.md"
    record = str(out / OZONE.name)
    assert main(["records-check", record, "--folder", str(folder), "--summary", str(summary)]) == 0
    assert summary.read_text(encoding="utf-8").startswith(f"- {OZONE.name}: 404 cells equal")
