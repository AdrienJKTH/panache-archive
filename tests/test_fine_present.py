import json
from datetime import date
from pathlib import Path

import pytest

from panache_archive import cli, fine_present
from panache_archive.fine_present import Span

NO2 = "nitrogen_dioxide"
CATCH_UP = [
    f"downscaled-{NO2}-present-2026-10-01-to-2026-10-31.days",
    f"downscaled-{NO2}-present-2026-11-01-to-2026-11-27.days",
    "downscaled-ozone-present-2026-11-01-to-2026-11-27.days",
    "downscaling-present-ozone.lgb.txt",
]


def test_the_days_after_the_last_file_up_to_yesterday_are_missing() -> None:
    found = fine_present.missing(CATCH_UP, date(2026, 11, 29))

    assert [s.name for s in found] == [
        f"downscaled-{NO2}-present-2026-11-28-to-2026-11-28.days",
        "downscaled-ozone-present-2026-11-28-to-2026-11-28.days",
    ]
    assert fine_present.missing(CATCH_UP, date(2026, 11, 28)) == []
    assert fine_present.missing(["downscaling-present-ozone.lgb.txt"], date(2026, 12, 1)) == []


def test_missed_days_across_a_month_s_end_make_a_file_a_month() -> None:
    found = fine_present.missing(CATCH_UP[2:], date(2026, 12, 3))

    assert found == [
        Span("ozone", date(2026, 11, 28), date(2026, 11, 30)),
        Span("ozone", date(2026, 12, 1), date(2026, 12, 2)),
    ]


def test_a_month_over_and_whole_in_several_files_is_joined() -> None:
    names = [
        *CATCH_UP,
        f"downscaled-{NO2}-present-2026-11-28-to-2026-11-29.days",
        f"downscaled-{NO2}-present-2026-11-30-to-2026-11-30.days",
    ]

    (join,) = fine_present.joins(names, date(2026, 12, 1))

    assert join.whole.name == f"downscaled-{NO2}-present-2026-11-01-to-2026-11-30.days"
    assert [p.first.day for p in join.parts] == [1, 28, 30]
    # Ozone's November lacks its last days; October is one file already; November not over yet.
    assert fine_present.joins(names, date(2026, 11, 30)) == []


def test_a_month_with_a_hole_or_without_its_start_is_left_alone() -> None:
    hole = [
        "downscaled-ozone-present-2026-11-01-to-2026-11-10.days",
        "downscaled-ozone-present-2026-11-12-to-2026-11-30.days",
    ]
    late = [
        "downscaled-ozone-present-2026-11-02-to-2026-11-10.days",
        "downscaled-ozone-present-2026-11-11-to-2026-11-30.days",
    ]

    assert fine_present.joins(hole, date(2026, 12, 5)) == []
    assert fine_present.joins(late, date(2026, 12, 5)) == []


def test_the_cli_says_what_to_map_and_what_to_join(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = tmp_path / "files.txt"
    files.write_text(
        "\n".join([*CATCH_UP, f"downscaled-{NO2}-present-2026-11-28-to-2026-11-30.days"]),
        encoding="utf-8",
    )

    assert cli.main(["present-plan", "--files", str(files), "--today", "2026-11-29"]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned == [{"pollutant": "ozone", "first": "2026-11-28", "last": "2026-11-28"}]

    assert cli.main(["present-joins", "--files", str(files), "--today", "2026-12-01"]) == 0
    (join,) = json.loads(capsys.readouterr().out)
    assert join["pollutant"] == NO2
    assert join["name"] == f"downscaled-{NO2}-present-2026-11-01-to-2026-11-30.days"
    assert join["parts"] == [
        f"downscaled-{NO2}-present-2026-11-01-to-2026-11-27.days",
        f"downscaled-{NO2}-present-2026-11-28-to-2026-11-30.days",
    ]


def test_a_forecast_is_made_once_a_run_and_older_runs_are_dropped() -> None:
    names = [
        *CATCH_UP,
        "downscaled-ozone-forecast-2026-11-26-to-2026-11-29.days",
        "downscaled-ozone-forecast-2026-11-27-to-2026-11-30.days",
        "downscaled-ozone-forecast-2026-11-28-to-2026-12-01.days",
    ]

    assert fine_present.without_forecast(names, date(2026, 11, 28)) == [NO2]
    assert fine_present.stale_forecasts(names, date(2026, 11, 28)) == [
        "downscaled-ozone-forecast-2026-11-26-to-2026-11-29.days"
    ]
    # A forecast is not a day mapped from the stream: yesterday is still missing.
    assert [s.first for s in fine_present.missing(names, date(2026, 11, 29))][-1] == date(
        2026, 11, 28
    )


def test_the_cli_says_which_forecasts_to_make_and_drop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = tmp_path / "files.txt"
    old = "downscaled-ozone-forecast-2026-11-20-to-2026-11-23.days"
    files.write_text("\n".join([*CATCH_UP, old]), encoding="utf-8")

    assert cli.main(["present-forecasts", "--files", str(files), "--today", "2026-11-28"]) == 0
    assert json.loads(capsys.readouterr().out) == {"build": [NO2, "ozone"], "stale": [old]}
