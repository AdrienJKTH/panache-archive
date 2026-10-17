"""Command line: `panache-archive plan`, `build`, `check`, `supervise`, `era5-plan` and
`era5-build`, as the workflows run them."""

import argparse
import asyncio
import io
import json
import logging
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

from panache_archive import ads, check, era5, plan, supervise
from panache_archive.build import ALL_MONTHS, Job, build, read_metadata
from panache_archive.variables import Reanalysis, Variable


def main(argv: Sequence[str] | None = None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")  # reports hold ✓ and μ, beyond Windows' cp1252
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parser().parse_args(argv)
    code: int = args.run(args)
    return code


def _plan(args: argparse.Namespace) -> int:
    built = set(Path(args.built).read_text(encoding="utf-8").split()) if args.built else set()
    skipped = {name.strip() for name in args.skip.split(",") if name.strip()}
    jobs = plan.wanted(
        plan.parse_variables(args.variables),
        plan.parse_years(args.years),
        ads.complete_years,
    )
    print(plan.matrix(plan.missing(jobs, built | skipped)))
    return 0


def _build(args: argparse.Namespace) -> int:
    job = Job(
        variable=Variable(args.variable),
        year=args.year,
        stream=Reanalysis(args.stream),
        months=_months(args.months),
    )
    with tempfile.TemporaryDirectory(dir=args.workdir) as workdir:
        path = build(job, Path(args.out), Path(workdir), download=ads.download)
    print(path)
    return 0


def _check(args: argparse.Namespace) -> int:
    agree = True
    for asset in args.assets:
        path = Path(asset)
        if read_metadata(path).get("source") == "era5":
            comparisons = era5.compare(path, get=check.http_get)
        else:
            comparisons = check.compare(path, get=check.http_get)
        report = check.render(path.name, comparisons)
        print(report)
        if args.summary:
            with Path(args.summary).open("a", encoding="utf-8") as summary:
                summary.write(report + "\n")
        agree = agree and all(c.agrees for c in comparisons)
    return 0 if agree else 1


def _supervise(args: argparse.Namespace) -> int:
    decision = supervise.supervise(
        Path(args.state),
        Path(args.out),
        at=datetime.now(UTC),
        supervisor_url=args.url,
        offered=ads.complete_years,
    )
    print(decision)
    return 0


def _era5_plan(args: argparse.Namespace) -> int:
    built = set(Path(args.built).read_text(encoding="utf-8").split()) if args.built else set()
    last = era5.last_complete_year(date.today())
    years = plan.parse_years(args.years)
    wanted = sorted(y for y in (years or range(era5.FIRST_YEAR, last + 1)) if y <= last)
    jobs = era5.missing_years(wanted, built)
    print(json.dumps([{"asset": job.asset_name, "year": job.year} for job in jobs]))
    return 0


def _era5_build(args: argparse.Namespace) -> int:
    job = era5.Era5Job(year=args.year, months=_months(args.months))
    path = asyncio.run(_era5_build_with_bucket(job, Path(args.out)))
    print(path)
    return 0


async def _era5_build_with_bucket(job: era5.Era5Job, out: Path) -> Path:
    async with era5.HttpBucket() as bucket:
        return await era5.build(job, out, bucket)


def _months(text: str) -> tuple[int, ...]:
    """`all`, a month such as `2`, or a span such as `2-5`."""
    return ALL_MONTHS if text.strip() == "all" else plan.parse_span(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panache-archive",
        description="Daily air quality over France from the CAMS European reanalyses.",
    )
    commands = parser.add_subparsers(required=True)

    planning = commands.add_parser("plan", help="print the matrix of the assets still missing")
    planning.add_argument("--variables", default="all", help="`all`, or names such as pm10,ozone")
    planning.add_argument("--years", default="all", help="`all`, a year, or a span: 2013-2025")
    planning.add_argument("--built", help="a file listing the assets already built")
    planning.add_argument("--skip", default="", help="assets to leave out, separated by commas")
    planning.set_defaults(run=_plan)

    building = commands.add_parser("build", help="build one asset from the ADS")
    building.add_argument("--variable", required=True, choices=[v.value for v in Variable])
    building.add_argument("--year", required=True, type=int)
    building.add_argument(
        "--stream", default=Reanalysis.VALIDATED.value, choices=[s.value for s in Reanalysis]
    )
    building.add_argument("--months", default="all", help="`all`, a month, or a span: 2-5")
    building.add_argument("--out", default="dist", help="where the asset is written")
    building.add_argument("--workdir", help="where downloads go (one month at a time)")
    building.set_defaults(run=_build)

    checking = commands.add_parser("check", help="compare assets with Open-Meteo")
    checking.add_argument("assets", nargs="+")
    checking.add_argument("--summary", help="a Markdown file to append the reports to")
    checking.set_defaults(run=_check)

    supervising = commands.add_parser("supervise", help="decide the next step, write the report")
    supervising.add_argument("--state", required=True, help="what the workflow gathered")
    supervising.add_argument("--out", required=True, help="where the decision is written")
    supervising.add_argument("--url", default="", help="this supervisor run's page")
    supervising.set_defaults(run=_supervise)

    era5_planning = commands.add_parser("era5-plan", help="print the ERA5 years still missing")
    era5_planning.add_argument("--years", default="all", help="`all`, a year, or a span")
    era5_planning.add_argument("--built", help="a file listing the assets already built")
    era5_planning.set_defaults(run=_era5_plan)

    era5_building = commands.add_parser("era5-build", help="build one ERA5 year from the bucket")
    era5_building.add_argument("--year", required=True, type=int)
    era5_building.add_argument("--months", default="all", help="`all`, a month, or a span: 2-5")
    era5_building.add_argument("--out", default="dist", help="where the asset is written")
    era5_building.set_defaults(run=_era5_build)
    return parser


if __name__ == "__main__":
    sys.exit(main())
