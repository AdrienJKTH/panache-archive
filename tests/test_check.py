from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest

from helpers import FakeAds, FakeOpenMeteo, open_meteo_answer
from panache_archive.build import Job, build
from panache_archive.check import (
    PLACES,
    Place,
    compare,
    open_meteo_hourly,
    render,
    rounding_step,
    tolerance,
)
from panache_archive.variables import Reanalysis, Variable

FEBRUARY = datetime(2021, 2, 1, tzinfo=UTC)
LYON = Place("Lyon", 45.76, 4.84)


@pytest.fixture
def asset(fake_ads: FakeAds, tmp_path: Path) -> Path:
    job = Job(Variable.OZONE, 2021, Reanalysis.VALIDATED, (2,))
    return build(job, tmp_path / "dist", tmp_path / "work", download=fake_ads)


def test_an_asset_agrees_with_open_meteo(asset: Path) -> None:
    open_meteo = FakeOpenMeteo(Variable.OZONE)

    comparisons = compare(asset, [LYON], get=open_meteo)

    assert [c.column for c in comparisons] == ["mean", "max_8h_mean"]
    for c in comparisons:
        assert c.cell == (45.75, 4.85)
        assert c.days == 28
        assert c.max_difference < 0.06
        assert c.correlation > 0.999
        assert c.agrees
    assert open_meteo.queries == [
        {
            "latitude": "45.76",
            "longitude": "4.84",
            "hourly": "ozone",
            "domains": "cams_europe",
            "timezone": "GMT",
            "start_date": "2021-02-01",
            "end_date": "2021-02-28",
        }
    ]


def test_an_offset_beyond_the_tolerance_disagrees(asset: Path) -> None:
    comparisons = compare(asset, [LYON], get=FakeOpenMeteo(Variable.OZONE, offset=0.2))

    assert not any(c.agrees for c in comparisons)


def test_the_report_is_a_markdown_table(asset: Path) -> None:
    comparisons = compare(asset, [LYON], get=FakeOpenMeteo(Variable.OZONE))

    report = render(asset.name, comparisons)

    assert report.startswith("### `cams-ozone-2021-02-validated.parquet` against Open-Meteo\n")
    assert "| Lyon | 45.75, 4.85 | mean | 28 |" in report
    assert "| 0.055 |" in report
    assert report.count("✓") == 2


def test_the_default_places_lie_in_the_box() -> None:
    assert [p.name for p in PLACES] == ["Paris", "Lyon", "Grenoble", "Bordeaux"]
    assert all(41.0 < p.lat < 51.5 and -5.5 < p.lon < 10.0 for p in PLACES)


def test_open_meteo_must_answer_whole_days() -> None:
    def short(_url: str) -> bytes:
        return open_meteo_answer(Variable.PM10, FEBRUARY, np.ones(30))

    with pytest.raises(ValueError, match="answered 30 hours, expected 48"):
        open_meteo_hourly(Variable.PM10, LYON, date(2021, 2, 1), date(2021, 2, 2), short)


def test_nulls_from_open_meteo_read_as_nan() -> None:
    values = np.ones(24)
    values[3] = np.nan

    def answer(_url: str) -> bytes:
        return open_meteo_answer(Variable.PM10, FEBRUARY, values)

    hourly = open_meteo_hourly(Variable.PM10, LYON, date(2021, 2, 1), date(2021, 2, 1), answer)

    assert np.isnan(hourly[3])
    assert np.nansum(hourly) == 23


def test_ozone_rounded_to_whole_units_still_agrees(asset: Path) -> None:
    comparisons = compare(asset, [LYON], get=FakeOpenMeteo(Variable.OZONE, decimals=0))

    for c in comparisons:
        assert c.tolerance == pytest.approx(0.505)
        assert 0.06 < c.max_difference <= 0.505
        assert c.agrees


@pytest.mark.parametrize(
    ("values", "step"),
    [
        ([61.0, 62.0, np.nan, 0.0], 1.0),
        ([61.0, 62.3, 0.1], 0.1),
        ([0.01, 0.25, 3.0], 0.01),
        ([0.123, 1.0], 0.0),
        ([np.nan], 1.0),
    ],
)
def test_the_rounding_step_is_read_from_the_values(values: list[float], step: float) -> None:
    assert rounding_step(np.array(values)) == step


def test_the_tolerance_is_half_a_step_plus_the_archive_rounding() -> None:
    assert tolerance(0.1) == pytest.approx(0.055)
    assert tolerance(1.0) == pytest.approx(0.505)
