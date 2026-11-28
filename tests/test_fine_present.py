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
