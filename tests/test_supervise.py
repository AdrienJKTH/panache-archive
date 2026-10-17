import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from panache_archive.build import Job
from panache_archive.supervise import (
    CHECK_STEP,
    MAX_FAILURES,
    Assessment,
    JobRecord,
    Run,
    Snapshot,
    Status,
    assess,
    events,
    job_asset,
    parse_jobs,
    parse_runs,
    render,
    supervise,
)
from panache_archive.variables import Reanalysis, Variable

AT = datetime(2026, 10, 17, 14, 0, tzinfo=UTC)
PM10_2013 = Job(Variable.PM10, 2013, Reanalysis.VALIDATED)
PM10_2014 = Job(Variable.PM10, 2014, Reanalysis.VALIDATED)
PM10_2025 = Job(Variable.PM10, 2025, Reanalysis.INTERIM)
DUST_2024 = Job(Variable.DUST, 2024, Reanalysis.VALIDATED)
WANTED = (PM10_2013, PM10_2014, PM10_2025, DUST_2024)


def run(number: int, status: str = "completed", conclusion: str | None = "success") -> Run:
    return Run(
        id=number * 100,
        number=number,
        status=status,
        conclusion=conclusion,
        url=f"https://github.com/runs/{number}",
    )


def job(
    run_number: int,
    name: str,
    status: str = "completed",
    conclusion: str | None = "success",
    minutes: float = 20.0,
    failed_step: str | None = None,
) -> JobRecord:
    started = AT - timedelta(hours=3)
    done = status == "completed"
    return JobRecord(
        run_id=run_number * 100,
        name=name,
        status=status,
        conclusion=conclusion if done else None,
        started_at=started if status != "queued" else None,
        completed_at=started + timedelta(minutes=minutes) if done else None,
        failed_step=failed_step,
    )


def test_statuses_follow_the_release_and_the_runs() -> None:
    runs = [run(2, "in_progress", None), run(1, conclusion="failure")]
    jobs = [
        job(1, PM10_2013.asset_name),
        job(1, PM10_2014.asset_name, conclusion="failure", failed_step="Build"),
        job(1, DUST_2024.asset_name, conclusion="failure", failed_step=CHECK_STEP),
        job(2, PM10_2014.asset_name, "in_progress"),
        job(2, PM10_2025.asset_name, "queued"),
    ]
    built = {PM10_2013.asset_name: 9_000_000, DUST_2024.asset_name: 7_000_000}

    now = assess(WANTED, built, runs, jobs)

    assert now.statuses == {
        PM10_2013.asset_name: Status.BUILT,
        PM10_2014.asset_name: Status.BUILDING,
        PM10_2025.asset_name: Status.QUEUED,
        DUST_2024.asset_name: Status.DISAGREES,
    }
    assert now.active_run == runs[0]
    assert now.built_bytes == 16_000_000
    assert not now.dispatch  # a run is on
    assert now.left == 2


def test_failed_assets_are_tried_again_then_left_out() -> None:
    runs = [run(n, conclusion="failure") for n in (3, 2, 1)]
    jobs = [
        *(job(n, PM10_2013.asset_name, conclusion="failure") for n in (1, 2, 3)),
        job(3, PM10_2014.asset_name, conclusion="timed_out"),
        job(3, PM10_2025.asset_name, conclusion="cancelled"),
    ]

    now = assess(WANTED, {}, runs, jobs)

    assert now.statuses[PM10_2013.asset_name] is Status.STUCK
    assert now.failures[PM10_2013.asset_name] == MAX_FAILURES
    assert now.statuses[PM10_2014.asset_name] is Status.RETRY
    assert now.statuses[PM10_2025.asset_name] is Status.MISSING  # cancelled is no failure
    assert now.dispatch
    assert now.stuck == {PM10_2013.asset_name}


def test_nothing_is_dispatched_once_complete() -> None:
    built = {j.asset_name: 1 for j in WANTED}

    now = assess(WANTED, built, [run(1)], [])

    assert now.complete
    assert not now.dispatch
    assert now.eta is None


def test_old_job_names_are_read_by_variable_and_year() -> None:
    by_key = {("pm10", 2013): PM10_2013.asset_name}

    assert job_asset("pm10 2013", by_key) == PM10_2013.asset_name
    assert job_asset("cams-dust-2024-validated.parquet", by_key) == DUST_2024.asset_name
    assert job_asset("Plan", by_key) is None
    assert job_asset("ozone 2013", by_key) is None


def test_the_pace_comes_from_successful_build_jobs() -> None:
    jobs = [
        job(1, PM10_2013.asset_name, minutes=10),
        job(1, DUST_2024.asset_name, minutes=30),
        job(1, "Plan", minutes=1),
    ]
    built = {PM10_2013.asset_name: 1, DUST_2024.asset_name: 1}

    now = assess(WANTED, built, [run(1)], jobs)

    assert now.job_minutes == 20
    assert now.eta == timedelta(minutes=20 * 2 / 4)


def test_interim_assets_are_superseded_by_validated_twins() -> None:
    built = {
        "cams-pm10-2024-interim.parquet": 1,
        "cams-pm10-2024-validated.parquet": 1,
        "cams-pm10-2025-interim.parquet": 1,
    }

    now = assess(WANTED, built, [], [])

    assert now.superseded == ("cams-pm10-2024-interim.parquet",)


def test_the_snapshot_survives_the_issue_body() -> None:
    snapshot = Snapshot(
        active_run=4,
        stuck=frozenset({"a"}),
        disagreeing=frozenset({"b"}),
        complete=False,
        wanted=(PM10_2013, PM10_2025),
        wanted_at=AT,
    )

    body = f"## Report\n\n<!-- supervisor-state {snapshot.to_json()} -->\n"

    assert Snapshot.from_body(body) == snapshot
    assert Snapshot.from_body("no state here") == Snapshot()
    assert Snapshot.from_body(None) == Snapshot()


def assessment(**statuses: Status) -> Assessment:
    names = {j.variable.value + str(j.year): j for j in WANTED}
    return Assessment(
        wanted=WANTED,
        statuses={j.asset_name: statuses.get(key, Status.BUILT) for key, j in names.items()},
        failures={},
        active_run=None,
        built_bytes=30_000_000,
        job_minutes=20.0,
        superseded=(),
    )


def test_a_run_that_ended_is_reported_with_its_counts() -> None:
    ended = run(4, conclusion="failure")
    jobs = [
        job(4, PM10_2013.asset_name),
        job(4, DUST_2024.asset_name, conclusion="failure", failed_step=CHECK_STEP),
        job(4, PM10_2014.asset_name, conclusion="failure", failed_step="Build"),
        job(4, "Plan"),
    ]
    now = assessment(pm102014=Status.RETRY)

    comments = events(Snapshot(active_run=ended.id), now, [ended], jobs)

    assert comments[0] == (
        "🏁 Run [#4](https://github.com/runs/4) ended (failure): 2 assets built, 1 failed. "
        "1 assets left."
    )
    assert comments[1].startswith("▶️ Launched a CAMS run for 1 assets, 1 of them to try again")


def test_new_stuck_and_disagreeing_assets_are_reported_once() -> None:
    now = assessment(pm102013=Status.STUCK, dust2024=Status.DISAGREES)

    first = events(Snapshot(), now, [], [])
    again = events(Snapshot(stuck=now.stuck, disagreeing=now.disagreeing), now, [], [])

    assert first[0].startswith("⛔ Failed 3 times")
    assert "`cams-pm10-2013-validated.parquet`" in first[0]
    assert first[1].startswith("⚠️ Built, but their check against Open-Meteo failed")
    assert again == []


def test_completion_is_announced_once() -> None:
    now = assessment()

    assert events(Snapshot(), now, [], []) == ["🎉 The archive is complete: 4 assets, 30 MB."]
    assert events(Snapshot(complete=True), now, [], []) == []


def test_the_report_holds_a_grid_and_the_snapshot() -> None:
    now = assessment(pm102014=Status.BUILDING, pm102025=Status.MISSING)
    snapshot = Snapshot(active_run=7)

    body = render(now, snapshot, AT, "https://github.com/supervisor")

    assert body.startswith("## CAMS archive: 2 of 4 assets\n")
    assert "`██████████░░░░░░░░░░` 50%" in body
    assert "**2 built** (30 MB) · 1 building · 1 not started" in body
    assert "about 0 h 10 min left" in body
    assert "| | 2013 | 2014 | 2024 | 2025 |" in body
    assert "| `pm10` | ✅ | 🔄 |  | ▫️ |" in body
    assert "| `dust` |  |  | ✅ |  |" in body
    assert "Updated 2026-10-17 14:00 UTC by [the supervisor](https://github.com/supervisor)" in body
    assert Snapshot.from_body(body) == snapshot


def test_the_report_says_when_nothing_is_left() -> None:
    body = render(assessment(), Snapshot(), AT, "")

    assert "Nothing left to build." in body


def github_runs(*runs: Run) -> list[dict[str, Any]]:
    return [
        {
            "id": r.id,
            "run_number": r.number,
            "status": r.status,
            "conclusion": r.conclusion,
            "html_url": r.url,
        }
        for r in runs
    ]


def test_runs_and_jobs_are_read_from_the_github_api() -> None:
    runs = parse_runs(github_runs(run(1)))
    jobs = parse_jobs(
        [
            {
                "run_id": 100,
                "name": "pm10 2013",
                "status": "completed",
                "conclusion": "failure",
                "started_at": "2026-10-17T13:51:22Z",
                "completed_at": "2026-10-17T14:11:22Z",
                "failed_step": "Build",
            }
        ]
    )

    assert runs == [run(1)]
    assert jobs[0].minutes == 20
    assert jobs[0].failed_step == "Build"


def write_state(
    state: Path,
    built: dict[str, int],
    runs: list[Run],
    jobs: list[dict[str, Any]],
    body: str | None,
) -> None:
    state.mkdir()
    (state / "built.json").write_text(
        json.dumps([{"name": n, "size": s} for n, s in built.items()]), encoding="utf-8"
    )
    (state / "runs.json").write_text(json.dumps(github_runs(*runs)), encoding="utf-8")
    (state / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    issues = [] if body is None else [{"number": 1, "body": body}]
    (state / "issue.json").write_text(json.dumps(issues), encoding="utf-8")


def offered(variable: Variable, stream: Reanalysis) -> frozenset[int]:
    if variable is not Variable.PM10:
        return frozenset()
    return frozenset({2013, 2014}) if stream is Reanalysis.VALIDATED else frozenset()


def test_a_first_supervision_asks_the_ads_and_launches_a_run(tmp_path: Path) -> None:
    write_state(tmp_path / "state", {}, [], [], None)

    decision = supervise(
        tmp_path / "state", tmp_path / "out", at=AT, supervisor_url="u", offered=offered
    )

    assert decision.dispatch
    assert json.loads((tmp_path / "out" / "decision.json").read_text(encoding="utf-8")) == {
        "dispatch": True,
        "skip": [],
        "delete": [],
    }
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert report.startswith("## CAMS archive: 0 of 2 assets")
    comments = sorted((tmp_path / "out" / "comments").iterdir())
    assert [c.read_text(encoding="utf-8") for c in comments] == [
        "▶️ Launched a CAMS run for 2 assets.\n"
    ]


def test_the_ads_is_asked_again_only_once_a_day(tmp_path: Path) -> None:
    kept = Snapshot(wanted=(PM10_2013,), wanted_at=AT - timedelta(hours=2), active_run=100)
    write_state(
        tmp_path / "state",
        {PM10_2013.asset_name: 5},
        [run(1)],
        [],
        render(assessment(), kept, AT, ""),
    )

    def refuse(_variable: Variable, _stream: Reanalysis) -> frozenset[int]:
        raise AssertionError("the ADS was asked again")

    decision = supervise(
        tmp_path / "state", tmp_path / "out", at=AT, supervisor_url="u", offered=refuse
    )

    assert not decision.dispatch
    comments = [
        c.read_text(encoding="utf-8") for c in sorted((tmp_path / "out" / "comments").iterdir())
    ]
    assert comments[0].startswith("🏁 Run [#1]")
    assert comments[1] == "🎉 The archive is complete: 1 assets, 0 MB.\n"


@pytest.mark.parametrize("age_hours", [25, 100])
def test_a_stale_list_is_asked_again(tmp_path: Path, age_hours: int) -> None:
    kept = Snapshot(wanted=(PM10_2013,), wanted_at=AT - timedelta(hours=age_hours))
    write_state(tmp_path / "state", {}, [], [], render(assessment(), kept, AT, ""))

    supervise(tmp_path / "state", tmp_path / "out", at=AT, supervisor_url="u", offered=offered)

    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "0 of 2 assets" in report
