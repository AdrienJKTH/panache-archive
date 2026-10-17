import json
from pathlib import Path

import pytest

from helpers import FakeAds, FakeOpenMeteo
from panache_archive import cli
from panache_archive.build import Job
from panache_archive.check import PLACES
from panache_archive.variables import Reanalysis, Variable


def test_plan_prints_the_missing_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def complete_years(variable: Variable, stream: Reanalysis) -> frozenset[int]:
        if variable is not Variable.PM10:
            return frozenset()
        return frozenset({2023, 2024}) if stream is Reanalysis.VALIDATED else frozenset({2025})

    monkeypatch.setattr("panache_archive.ads.complete_years", complete_years)
    built = tmp_path / "built.txt"
    built.write_text("cams-pm10-2023-validated.parquet\nother.parquet\n", encoding="utf-8")

    asked = ["plan", "--variables", "pm10,dust", "--years", "2023-2025", "--built", str(built)]

    code = cli.main([*asked, "--skip", "cams-pm10-2025-interim.parquet,"])

    assert code == 0
    assert json.loads(capsys.readouterr().out) == [
        {
            "asset": "cams-pm10-2024-validated.parquet",
            "variable": "pm10",
            "year": 2024,
            "stream": "validated_reanalysis",
        }
    ]


def test_build_writes_the_asset(
    monkeypatch: pytest.MonkeyPatch,
    fake_ads: FakeAds,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("panache_archive.ads.download", fake_ads)
    out = tmp_path / "dist"
    workdir = tmp_path / "work"
    workdir.mkdir()

    asked = ["build", "--variable", "dust", "--year", "2021", "--months", "2"]

    code = cli.main([*asked, "--out", str(out), "--workdir", str(workdir)])

    assert code == 0
    assert capsys.readouterr().out.strip() == str(out / "cams-dust-2021-02-validated.parquet")
    assert list(workdir.iterdir()) == []


def test_a_whole_year_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs: list[Job] = []

    def build(job: Job, *_: object, **__: object) -> Path:
        jobs.append(job)
        return Path("built.parquet")

    monkeypatch.setattr(cli, "build", build)

    cli.main(["build", "--variable", "pm10", "--year", "2021", "--stream", "interim_reanalysis"])

    (job,) = jobs
    assert job.months == tuple(range(1, 13))
    assert job.stream is Reanalysis.INTERIM


@pytest.mark.parametrize(("offset", "expected"), [(0.0, 0), (1.0, 1)])
def test_check_reports_and_fails_on_disagreement(
    monkeypatch: pytest.MonkeyPatch,
    fake_ads: FakeAds,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    offset: float,
    expected: int,
) -> None:
    monkeypatch.setattr("panache_archive.ads.download", fake_ads)
    cli.main(
        ["build", "--variable", "pm10", "--year", "2021", "--months", "2", "--out", str(tmp_path)]
    )
    asset = tmp_path / "cams-pm10-2021-02-validated.parquet"
    capsys.readouterr()

    monkeypatch.setattr("panache_archive.check.http_get", FakeOpenMeteo(Variable.PM10, offset))
    summary = tmp_path / "summary.md"

    code = cli.main(["check", str(asset), "--summary", str(summary)])

    assert code == expected
    report = summary.read_text(encoding="utf-8")
    assert report.count("| mean |") == len(PLACES)
    assert report in capsys.readouterr().out + "\n"


def test_supervise_writes_its_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def complete_years(variable: Variable, stream: Reanalysis) -> frozenset[int]:
        offered = variable is Variable.PM10 and stream is Reanalysis.VALIDATED
        return frozenset({2013}) if offered else frozenset()

    monkeypatch.setattr("panache_archive.ads.complete_years", complete_years)
    state = tmp_path / "state"
    state.mkdir()
    for name in ("built", "runs", "jobs", "issue"):
        (state / f"{name}.json").write_text("[]", encoding="utf-8")

    code = cli.main(["supervise", "--state", str(state), "--out", str(tmp_path / "out")])

    assert code == 0
    assert "dispatch=True" in capsys.readouterr().out
    decision = json.loads((tmp_path / "out" / "decision.json").read_text(encoding="utf-8"))
    assert decision == {"dispatch": True, "skip": [], "delete": []}
