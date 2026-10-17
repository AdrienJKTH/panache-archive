"""The supervisor: it watches the CAMS runs, launches them again until the archive is whole, and
reports progress in an issue.

The `Supervise` workflow runs it after each CAMS build job, whenever a CAMS run ends, and every
15 minutes. Each time, it
reads the release, the recent CAMS runs with their jobs, and its own last snapshot, kept in the
issue's body. Then it decides:
- to dispatch a CAMS run when assets are missing and none runs (a run builds only what the
  release lacks);
- to leave out an asset that failed `MAX_FAILURES` times, until someone looks;
- to delete an interim asset once its validated twin is built.
It writes the issue's body (every asset in a grid) and a comment for each change, which GitHub
notifies.
"""

import json
import re
import statistics
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from panache_archive import plan
from panache_archive.build import Job
from panache_archive.variables import Reanalysis, Variable

MAX_FAILURES: Final = 3
PARALLEL_JOBS: Final = 4
"""The `max-parallel` of the CAMS workflow's build jobs."""
WANTED_MAX_AGE: Final = timedelta(hours=24)
"""How long the list of assets the ADS offers is trusted before it is asked again."""
CHECK_STEP: Final = "Check against Open-Meteo"
ACTIVE: Final = frozenset({"queued", "in_progress", "waiting", "pending", "requested"})
FAILED: Final = frozenset({"failure", "timed_out"})
STATE_PATTERN: Final = re.compile(r"<!-- supervisor-state (.*?) -->", re.DOTALL)
OLD_JOB_NAME: Final = re.compile(r"([a-z0-9_]+) (\d{4})")
"""Build jobs were named `pm10 2013` before they were named after their asset."""


class Status(StrEnum):
    BUILT = "built"
    DISAGREES = "disagrees"
    BUILDING = "building"
    QUEUED = "queued"
    RETRY = "retry"
    STUCK = "stuck"
    MISSING = "missing"


SYMBOLS: Final[Mapping[Status, str]] = {
    Status.BUILT: "✅",
    Status.DISAGREES: "⚠️",
    Status.BUILDING: "🔄",
    Status.QUEUED: "⏳",
    Status.RETRY: "❌",
    Status.STUCK: "⛔",
    Status.MISSING: "▫️",
}
MEANINGS: Final[Mapping[Status, str]] = {
    Status.BUILT: "built",
    Status.DISAGREES: "built, but disagrees with Open-Meteo",
    Status.BUILDING: "building",
    Status.QUEUED: "queued",
    Status.RETRY: "failed, to try again",
    Status.STUCK: f"stuck after {MAX_FAILURES} failures, left out",
    Status.MISSING: "not built yet",
}
SHORT: Final[Mapping[Status, str]] = {
    Status.BUILDING: "building",
    Status.QUEUED: "queued",
    Status.RETRY: "to try again",
    Status.STUCK: "stuck",
    Status.MISSING: "not started",
}
"""How the summary line counts the assets not built yet."""
LEFT: Final = frozenset({Status.BUILDING, Status.QUEUED, Status.RETRY, Status.MISSING})


@dataclass(frozen=True, slots=True)
class Run:
    """A CAMS run, as the GitHub API lists it."""

    id: int
    number: int
    status: str
    conclusion: str | None
    url: str

    @property
    def active(self) -> bool:
        return self.status in ACTIVE


@dataclass(frozen=True, slots=True)
class JobRecord:
    """A job of a CAMS run, as the GitHub API lists it."""

    run_id: int
    name: str
    status: str
    conclusion: str | None
    started_at: datetime | None
    completed_at: datetime | None
    failed_step: str | None
    """The first step that failed, if any."""

    @property
    def minutes(self) -> float | None:
        if self.started_at is None or self.completed_at is None:
            return None
        return (self.completed_at - self.started_at) / timedelta(minutes=1)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """What the supervisor saw last time; it keeps it in the issue's body."""

    active_run: int | None = None
    stuck: frozenset[str] = frozenset()
    disagreeing: frozenset[str] = frozenset()
    complete: bool = False
    wanted: tuple[Job, ...] = ()
    wanted_at: datetime | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "active_run": self.active_run,
                "stuck": sorted(self.stuck),
                "disagreeing": sorted(self.disagreeing),
                "complete": self.complete,
                "wanted": [[j.variable.value, j.year, j.stream.value] for j in self.wanted],
                "wanted_at": self.wanted_at.isoformat() if self.wanted_at else None,
            }
        )

    @classmethod
    def from_body(cls, body: str | None) -> "Snapshot":
        """The snapshot kept in an issue's body; an empty one if there is none."""
        found = STATE_PATTERN.search(body or "")
        if found is None:
            return cls()
        data = json.loads(found.group(1))
        return cls(
            active_run=data.get("active_run"),
            stuck=frozenset(data.get("stuck", [])),
            disagreeing=frozenset(data.get("disagreeing", [])),
            complete=bool(data.get("complete", False)),
            wanted=tuple(
                Job(Variable(v), int(y), Reanalysis(s)) for v, y, s in data.get("wanted", [])
            ),
            wanted_at=_time(data.get("wanted_at")),
        )


@dataclass(frozen=True, slots=True)
class Assessment:
    """Where every asset stands, and what to do about it."""

    wanted: tuple[Job, ...]
    statuses: Mapping[str, Status]
    """By asset name, in the order of `wanted`."""
    failures: Mapping[str, int]
    active_run: Run | None
    built_bytes: int
    job_minutes: float | None
    """Median duration of the build jobs that succeeded."""
    superseded: tuple[str, ...]
    """Interim assets whose validated twin is built."""
    counts: Counter[Status] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", Counter(self.statuses.values()))

    @property
    def stuck(self) -> frozenset[str]:
        return frozenset(a for a, s in self.statuses.items() if s is Status.STUCK)

    @property
    def disagreeing(self) -> frozenset[str]:
        return frozenset(a for a, s in self.statuses.items() if s is Status.DISAGREES)

    @property
    def complete(self) -> bool:
        return bool(self.statuses) and all(
            s in (Status.BUILT, Status.DISAGREES) for s in self.statuses.values()
        )

    @property
    def dispatch(self) -> bool:
        """Launch a run: assets are missing or to try again, and no run is on."""
        waiting = self.counts[Status.MISSING] + self.counts[Status.RETRY]
        return self.active_run is None and waiting > 0

    @property
    def left(self) -> int:
        return sum(self.counts[s] for s in LEFT)

    @property
    def eta(self) -> timedelta | None:
        """Time left, at the median job's pace and `PARALLEL_JOBS` at once."""
        if self.job_minutes is None or self.left == 0:
            return None
        return timedelta(minutes=self.job_minutes * self.left / PARALLEL_JOBS)


def assess(
    wanted: Sequence[Job],
    built: Mapping[str, int],
    runs: Sequence[Run],
    jobs: Sequence[JobRecord],
) -> Assessment:
    """Each wanted asset's status, from the release (`built`: sizes by name) and the runs."""
    by_key = {(j.variable.value, j.year): j.asset_name for j in wanted}
    by_asset: dict[str, list[JobRecord]] = {}
    for record in jobs:
        asset = job_asset(record.name, by_key)
        if asset is not None:
            by_asset.setdefault(asset, []).append(record)
    active = next((run for run in runs if run.active), None)

    statuses: dict[str, Status] = {}
    failures: dict[str, int] = {}
    for job in wanted:
        records = by_asset.get(job.asset_name, [])
        failed = sum(r.conclusion in FAILED and r.failed_step != CHECK_STEP for r in records)
        failures[job.asset_name] = failed
        statuses[job.asset_name] = _status(job.asset_name in built, records, failed, active)

    durations = [
        minutes
        for records in by_asset.values()
        for r in records
        if r.conclusion == "success" and (minutes := r.minutes) is not None
    ]
    superseded = tuple(
        name
        for name in sorted(built)
        if name.endswith("-interim.parquet")
        and name.replace("-interim.parquet", "-validated.parquet") in built
    )
    return Assessment(
        wanted=tuple(wanted),
        statuses=statuses,
        failures=failures,
        active_run=active,
        built_bytes=sum(built.get(j.asset_name, 0) for j in wanted),
        job_minutes=statistics.median(durations) if durations else None,
        superseded=superseded,
    )


def job_asset(name: str, by_key: Mapping[tuple[str, int], str]) -> str | None:
    """The asset a job builds: its name, or its variable and year under the old names."""
    if name.endswith(".parquet"):
        return name
    old = OLD_JOB_NAME.fullmatch(name)
    return by_key.get((old.group(1), int(old.group(2)))) if old else None


def events(
    previous: Snapshot, now: Assessment, runs: Sequence[Run], jobs: Sequence[JobRecord]
) -> list[str]:
    """A comment for each change since the last snapshot, and for the run about to start."""
    comments: list[str] = []
    ended = next((r for r in runs if r.id == previous.active_run and not r.active), None)
    if ended is not None:
        comments.append(_ended(ended, now, jobs))
    if new := sorted(now.stuck - previous.stuck):
        comments.append(
            f"⛔ Failed {MAX_FAILURES} times, now left out of the runs until someone looks: "
            + _names(new)
        )
    if new := sorted(now.disagreeing - previous.disagreeing):
        comments.append(
            "⚠️ Built, but their check against Open-Meteo failed (see the job's summary): "
            + _names(new)
        )
    if now.superseded:
        comments.append(
            "🧹 Deleted interim assets, now that their validated twins are built: "
            + _names(now.superseded)
        )
    if now.dispatch:
        retry = now.counts[Status.RETRY]
        again = f", {retry} of them to try again" if retry else ""
        comments.append(
            f"▶️ Launched a CAMS run for {now.counts[Status.MISSING] + retry} assets{again}."
        )
    if now.complete and not previous.complete:
        comments.append(
            f"🎉 The archive is complete: {len(now.wanted)} assets, "
            f"{now.built_bytes / 1e6:,.0f} MB."
        )
    return comments


def render(now: Assessment, snapshot: Snapshot, at: datetime, supervisor_url: str) -> str:
    """The issue's body: a summary, a grid of every asset, the legend and the snapshot."""
    total = len(now.wanted)
    built = now.counts[Status.BUILT] + now.counts[Status.DISAGREES]
    share = built / total if total else 0.0
    parts = [f"{now.counts[s]} {label}" for s, label in SHORT.items() if now.counts[s]]
    lines = [
        f"## CAMS archive: {built} of {total} assets",
        "",
        f"`{_bar(share)}` {share:.0%}",
        "",
        f"**{built} built** ({now.built_bytes / 1e6:,.0f} MB)" + "".join(f" · {p}" for p in parts),
        "",
        _pace(now),
        "",
        *_grid(now),
        "",
        " · ".join(f"{SYMBOLS[s]} {MEANINGS[s]}" for s in Status)
        + " · blank: not offered by the ADS",
        "",
        f"Updated {at:%Y-%m-%d %H:%M} UTC by [the supervisor]({supervisor_url}), which runs "
        "after each asset and every 15 minutes.",
        "",
        f"<!-- supervisor-state {snapshot.to_json()} -->",
    ]
    return "\n".join(lines) + "\n"


def parse_runs(data: Sequence[Mapping[str, Any]]) -> list[Run]:
    return [
        Run(
            id=int(r["id"]),
            number=int(r["run_number"]),
            status=str(r["status"]),
            conclusion=r.get("conclusion"),
            url=str(r["html_url"]),
        )
        for r in data
    ]


def parse_jobs(data: Sequence[Mapping[str, Any]]) -> list[JobRecord]:
    return [
        JobRecord(
            run_id=int(j["run_id"]),
            name=str(j["name"]),
            status=str(j["status"]),
            conclusion=j.get("conclusion"),
            started_at=_time(j.get("started_at")),
            completed_at=_time(j.get("completed_at")),
            failed_step=j.get("failed_step"),
        )
        for j in data
    ]


@dataclass(frozen=True, slots=True)
class Decision:
    dispatch: bool
    skip: tuple[str, ...]
    delete: tuple[str, ...]


def supervise(
    state: Path,
    out: Path,
    *,
    at: datetime,
    supervisor_url: str,
    offered: plan.Years,
) -> Decision:
    """Read what the workflow gathered in `state`, and write the decision, the issue's body and
    the comments in `out`.

    `state` holds `built.json` (the release's assets: name and size), `runs.json` and
    `jobs.json` (recent CAMS runs and their jobs) and `issue.json` (the progress issue, if any).
    """
    built = {a["name"]: int(a["size"]) for a in _load(state / "built.json")}
    runs = parse_runs(_load(state / "runs.json"))
    jobs = parse_jobs(_load(state / "jobs.json"))
    issues = _load(state / "issue.json")
    previous = Snapshot.from_body(issues[0]["body"] if issues else None)

    wanted, wanted_at = previous.wanted, previous.wanted_at
    if not wanted or wanted_at is None or at - wanted_at > WANTED_MAX_AGE:
        wanted, wanted_at = tuple(plan.wanted(list(Variable), None, offered)), at
    now = assess(wanted, built, runs, jobs)
    comments = events(previous, now, runs, jobs)
    snapshot = Snapshot(
        active_run=now.active_run.id if now.active_run else None,
        stuck=now.stuck,
        disagreeing=now.disagreeing,
        complete=now.complete,
        wanted=wanted,
        wanted_at=wanted_at,
    )

    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render(now, snapshot, at, supervisor_url), encoding="utf-8")
    (out / "comments").mkdir(exist_ok=True)
    for index, comment in enumerate(comments, start=1):
        (out / "comments" / f"{index:02d}.md").write_text(comment + "\n", encoding="utf-8")
    decision = Decision(dispatch=now.dispatch, skip=tuple(sorted(now.stuck)), delete=now.superseded)
    (out / "decision.json").write_text(
        json.dumps(
            {
                "dispatch": decision.dispatch,
                "skip": list(decision.skip),
                "delete": list(decision.delete),
            }
        ),
        encoding="utf-8",
    )
    return decision


def _status(
    is_built: bool, records: Collection[JobRecord], failed: int, active: Run | None
) -> Status:
    if is_built:
        finished = [r for r in records if r.completed_at is not None]
        latest = max(finished, key=_completed) if finished else None
        disagrees = latest is not None and latest.failed_step == CHECK_STEP
        return Status.DISAGREES if disagrees else Status.BUILT
    current = next(
        (r for r in records if active is not None and r.run_id == active.id and r.status in ACTIVE),
        None,
    )
    if current is not None:
        return Status.BUILDING if current.status == "in_progress" else Status.QUEUED
    if failed >= MAX_FAILURES:
        return Status.STUCK
    return Status.RETRY if failed else Status.MISSING


def _completed(record: JobRecord) -> float:
    return record.completed_at.timestamp() if record.completed_at else 0.0


def _ended(run: Run, now: Assessment, jobs: Sequence[JobRecord]) -> str:
    assets = set(now.statuses)
    by_key = {(j.variable.value, j.year): j.asset_name for j in now.wanted}
    own = [j for j in jobs if j.run_id == run.id and job_asset(j.name, by_key) in assets]
    built = sum(j.conclusion == "success" or j.failed_step == CHECK_STEP for j in own)
    failed = sum(j.conclusion in FAILED and j.failed_step != CHECK_STEP for j in own)
    left = f" {now.left} assets left." if now.left else ""
    return (
        f"🏁 Run [#{run.number}]({run.url}) ended ({run.conclusion}): {built} assets built, "
        f"{failed} failed.{left}"
    )


def _pace(now: Assessment) -> str:
    if now.active_run is not None:
        run = f"Run [#{now.active_run.number}]({now.active_run.url}) is on."
    elif now.complete:
        return "Nothing left to build."
    else:
        run = "No run is on."
    if now.eta is None or now.job_minutes is None:
        return run
    hours, minutes = divmod(round(now.eta / timedelta(minutes=1)), 60)
    return (
        f"{run} At {now.job_minutes:.0f} min a job and {PARALLEL_JOBS} at a time, about "
        f"{hours} h {minutes:02d} min left."
    )


def _grid(now: Assessment) -> list[str]:
    years = sorted({j.year for j in now.wanted})
    cells = {(j.variable, j.year): SYMBOLS[now.statuses[j.asset_name]] for j in now.wanted}
    variables = [v for v in Variable if any(j.variable is v for j in now.wanted)]
    lines = [
        "| | " + " | ".join(str(y) for y in years) + " |",
        "|---|" + "---|" * len(years),
    ]
    for variable in variables:
        row = [cells.get((variable, year), "") for year in years]
        lines.append(f"| `{variable}` | " + " | ".join(row) + " |")
    return lines


def _bar(share: float, width: int = 20) -> str:
    filled = round(share * width)
    return "█" * filled + "░" * (width - filled)


def _names(assets: Sequence[str]) -> str:
    return ", ".join(f"`{a}`" for a in assets) + "."


def _time(text: str | None) -> datetime | None:
    return datetime.fromisoformat(text.replace("Z", "+00:00")) if text else None


def _load(path: Path) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return loaded
