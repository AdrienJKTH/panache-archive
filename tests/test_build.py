import zipfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

from helpers import GRID_LATS, GRID_LONS, FakeAds, synthetic
from panache_archive.build import ALL_MONTHS, Job, build, read_metadata, to_table
from panache_archive.daily import daily_max_8h_means, daily_means
from panache_archive.variables import Reanalysis, Statistic, Variable

LATS = np.array([41.05, 45.75, 51.45])
LONS = np.array([-5.45, 4.85, 9.95])


@pytest.mark.parametrize(
    ("months", "name"),
    [
        (ALL_MONTHS, "cams-pm10-2021-validated.parquet"),
        ((2,), "cams-pm10-2021-02-validated.parquet"),
        ((2, 3, 4), "cams-pm10-2021-02-04-validated.parquet"),
    ],
)
def test_asset_names(months: tuple[int, ...], name: str) -> None:
    assert Job(Variable.PM10, 2021, Reanalysis.VALIDATED, months).asset_name == name


def test_an_interim_asset_is_named_so() -> None:
    job = Job(Variable.DUST, 2025, Reanalysis.INTERIM)

    assert job.asset_name == "cams-dust-2025-interim.parquet"


@pytest.mark.parametrize("months", [(), (0,), (2, 4), (3, 2), (12, 13)])
def test_a_job_covers_consecutive_months_of_a_year(months: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="months"):
        Job(Variable.PM10, 2021, Reanalysis.VALIDATED, months)


def test_an_asset_holds_each_cell_and_day_sorted(fake_ads: FakeAds, tmp_path: Path) -> None:
    job = Job(Variable.PM10, 2021, Reanalysis.VALIDATED, (2,))

    path = build(job, tmp_path / "dist", tmp_path / "work", download=fake_ads)

    assert path == tmp_path / "dist" / "cams-pm10-2021-02-validated.parquet"
    table = pq.read_table(path)
    assert table.column_names == ["lat", "lon", "day", "mean"]
    assert table.num_rows == 3 * 3 * 28
    rows = table.to_pylist()
    assert rows[0]["lat"] == 41.05
    assert rows[0]["lon"] == -5.45
    assert rows[0]["day"] == date(2021, 2, 1)
    assert rows[27]["day"] == date(2021, 2, 28)
    assert (rows[28]["lat"], rows[28]["lon"]) == (41.05, 4.85)
    assert (rows[-1]["lat"], rows[-1]["lon"]) == (51.45, 9.95)
    assert fake_ads.requests == [
        {
            "variable": ["particulate_matter_10um"],
            "model": ["ensemble"],
            "level": ["0"],
            "type": ["validated_reanalysis"],
            "year": ["2021"],
            "month": ["02"],
        }
    ]
    assert list((tmp_path / "work").iterdir()) == []  # downloads are deleted once read


def test_daily_means_are_those_of_the_hours_rounded(fake_ads: FakeAds, tmp_path: Path) -> None:
    job = Job(Variable.PM10, 2021, Reanalysis.VALIDATED, (2,))

    table = pq.read_table(build(job, tmp_path / "dist", tmp_path / "work", download=fake_ads))

    hourly = synthetic(datetime(2021, 2, 1, tzinfo=UTC), 28 * 24, GRID_LATS, GRID_LONS)
    lyon = hourly.astype(np.float32).astype(np.float64)[:, 2, 2]  # 45.75, 4.85, as stored
    at_lyon = table.filter((table["lat"].to_numpy() == 45.75) & (table["lon"].to_numpy() == 4.85))
    stored = at_lyon["mean"].to_numpy()
    np.testing.assert_allclose(stored, np.round(daily_means(lyon), 2), atol=1e-4)


def test_ozone_carries_its_8_hour_maxima_across_months(fake_ads: FakeAds, tmp_path: Path) -> None:
    job = Job(Variable.OZONE, 2021, Reanalysis.VALIDATED, (1, 2))

    table = pq.read_table(build(job, tmp_path / "dist", tmp_path / "work", download=fake_ads))

    assert table.column_names == ["lat", "lon", "day", "mean", "max_8h_mean"]
    hourly = synthetic(datetime(2021, 1, 1, tzinfo=UTC), 59 * 24, GRID_LATS, GRID_LONS)
    cell = hourly.astype(np.float32).astype(np.float64)[:, 1, 1]  # 41.05, -5.45
    at_cell = table.filter((table["lat"].to_numpy() == 41.05) & (table["lon"].to_numpy() == -5.45))
    # 1 February's earliest means take 31 January's evening: as over one continuous series.
    np.testing.assert_allclose(
        at_cell["max_8h_mean"].to_numpy(), np.round(daily_max_8h_means(cell), 2), atol=1e-4
    )


def test_an_asset_says_what_it_holds(fake_ads: FakeAds, tmp_path: Path) -> None:
    job = Job(Variable.DUST, 2025, Reanalysis.INTERIM, (2,))

    metadata = read_metadata(build(job, tmp_path, tmp_path / "work", download=fake_ads))

    assert metadata["variable"] == "dust"
    assert metadata["unit"] == "μg/m³"
    assert metadata["statistics"] == ["mean"]
    assert metadata["stream"] == "interim_reanalysis"
    assert metadata["doi"] == "10.24381/7cc0465a"
    assert metadata["licence"] == "CC-BY-4.0"
    assert "Copernicus Atmosphere Monitoring Service" in metadata["attribution"]
    assert metadata["box"] == {"south": 41.0, "north": 51.5, "west": -5.5, "east": 10.0}


def test_a_plain_netcdf_download_works_too(fake_ads: FakeAds, tmp_path: Path) -> None:
    fake_ads.zipped = False
    job = Job(Variable.PM10, 2021, Reanalysis.VALIDATED, (2,))

    assert pq.read_table(build(job, tmp_path, tmp_path / "work", download=fake_ads)).num_rows == 252


def _shifted(month: int, options: dict[str, Any]) -> None:
    if month == 2:
        options["first_hour"] = datetime(2021, 2, 1, 1, tzinfo=UTC)


def _moved(month: int, options: dict[str, Any]) -> None:
    if month == 2:
        options["lats"] = GRID_LATS + 0.1


def _in_ppb(_month: int, options: dict[str, Any]) -> None:
    options["units"] = "ppb"


@pytest.mark.parametrize(
    ("alter", "error"),
    [
        (_shifted, "from 2021-02-01 01:00, expected 672 from 2021-02-01 00:00"),
        (_moved, "grid"),
        (_in_ppb, "ppb"),
    ],
)
def test_a_wrong_month_is_an_error(
    fake_ads: FakeAds, tmp_path: Path, alter: Any, error: str
) -> None:
    fake_ads.alter = alter
    job = Job(Variable.PM10, 2021, Reanalysis.VALIDATED, (1, 2))

    with pytest.raises(ValueError, match=error):
        build(job, tmp_path, tmp_path / "work", download=fake_ads)
    assert not any((tmp_path / "work").glob("*.nc"))


def test_a_month_in_several_files_is_an_error(tmp_path: Path) -> None:
    def two_files(_request: Any, target: Path) -> None:
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("a.nc", b"")
            archive.writestr("b.nc", b"")

    job = Job(Variable.PM10, 2021, Reanalysis.VALIDATED, (2,))

    with pytest.raises(ValueError, match="expected one netCDF file"):
        build(job, tmp_path, tmp_path / "work", download=two_files)
    assert not any((tmp_path / "work").iterdir())


def test_a_table_needs_columns_matching_the_cells() -> None:
    with pytest.raises(ValueError, match="do not match"):
        to_table(LATS, LONS, date(2021, 2, 1), {Statistic.MEAN: np.zeros((2, 3, 4))}, metadata={})


def test_missing_days_are_null() -> None:
    values = np.ones((2, 3, 3))
    values[1, 0, 0] = np.nan

    table = to_table(LATS, LONS, date(2021, 2, 1), {Statistic.MEAN: values}, metadata={})

    assert table["mean"].null_count == 1
    assert table["mean"][1].as_py() is None


def test_a_file_from_elsewhere_is_not_an_asset(tmp_path: Path) -> None:
    path = tmp_path / "other.parquet"
    pq.write_table(
        to_table(
            LATS, LONS, date(2021, 2, 1), {Statistic.MEAN: np.ones((1, 3, 3))}, metadata={}
        ).replace_schema_metadata(None),
        path,
    )

    with pytest.raises(ValueError, match="not an asset"):
        read_metadata(path)
