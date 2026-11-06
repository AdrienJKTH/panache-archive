"""Command line: `panache-archive plan`, `build`, `check`, `supervise`, `era5-plan`,
`era5-build`, `merra2-plan`, `merra2-build`, `mirror`, `stations`, `static`, `fine`,
`fine-plan` and `space`, as the
workflows run them."""

import argparse
import asyncio
import io
import json
import logging
import os
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

from huggingface_hub import HfApi

from panache_archive import (
    ads,
    check,
    era5,
    fine,
    merra2,
    mirror,
    plan,
    space,
    static,
    stations,
    supervise,
)
from panache_archive.build import ALL_MONTHS, Job, build, read_metadata
from panache_archive.domain import Domain
from panache_archive.grid import Box
from panache_archive.static import Grid
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
        Domain(args.domain),
    )
    print(plan.matrix(plan.missing(jobs, built | skipped)))
    return 0


def _build(args: argparse.Namespace) -> int:
    job = Job(
        variable=Variable(args.variable),
        year=args.year,
        stream=Reanalysis(args.stream),
        months=_months(args.months),
        domain=Domain(args.domain),
    )
    with tempfile.TemporaryDirectory(dir=args.workdir) as workdir:
        path = build(job, Path(args.out), Path(workdir), download=ads.download)
    print(path)
    return 0


def _check(args: argparse.Namespace) -> int:
    agree = True
    for asset in args.assets:
        path = Path(asset)
        source = read_metadata(path).get("source")
        reference = "Giovanni" if source == "merra2" else "Open-Meteo"
        if source == "era5":
            comparisons = era5.compare(path, get=check.http_get)
        elif source == "merra2":
            comparisons = merra2.compare(path, fetch=merra2.token_fetch(_earthdata_token()))
        else:
            comparisons = check.compare(path, get=check.http_get)
        report = check.render(path.name, comparisons, reference)
        agree = agree and all(c.agrees for c in comparisons)
        twin = Path(args.twins or ".") / path.with_suffix(Domain.FRANCE.suffix).name
        if args.twins and path.suffix == Domain.EUROPE.suffix and twin.exists():
            against = check.compare_twin(path, twin)
            report += "\n" + check.render_twin(path.name, against)
            agree = agree and against.agrees
        print(report)
        if args.summary:
            with Path(args.summary).open("a", encoding="utf-8") as summary:
                summary.write(report + "\n")
    return 0 if agree else 1


def _supervise(args: argparse.Namespace) -> int:
    decision = supervise.supervise(
        Path(args.state),
        Path(args.out),
        at=datetime.now(UTC),
        supervisor_url=args.url,
        offered=ads.complete_years,
        domain=Domain(args.domain),
    )
    print(decision)
    return 0


def _era5_plan(args: argparse.Namespace) -> int:
    built = set(Path(args.built).read_text(encoding="utf-8").split()) if args.built else set()
    end = era5.data_end(check.http_get)
    years = plan.parse_years(args.years)
    wanted = sorted(years or range(era5.FIRST_YEAR, end.year + 1))
    jobs, superseded = era5.plan(wanted, built, end, Domain(args.domain))
    print(
        json.dumps([{"asset": j.asset_name, "year": j.year, "months": j.months[-1]} for j in jobs])
    )
    if args.superseded:
        Path(args.superseded).write_text("\n".join(superseded), encoding="utf-8")
    return 0


def _era5_build(args: argparse.Namespace) -> int:
    job = era5.Era5Job(year=args.year, months=_months(args.months), domain=Domain(args.domain))
    path = asyncio.run(_era5_build_with_bucket(job, Path(args.out)))
    print(path)
    return 0


async def _era5_build_with_bucket(job: era5.Era5Job, out: Path) -> Path:
    async with era5.HttpBucket() as bucket:
        return await era5.build(job, out, bucket)


def _merra2_plan(args: argparse.Namespace) -> int:
    built = set(Path(args.built).read_text(encoding="utf-8").split()) if args.built else set()
    years = plan.parse_years(args.years)
    today = date.today()
    wanted = sorted(years or range(merra2.FIRST_YEAR, today.year + 1))
    fetch = merra2.token_fetch(_earthdata_token())
    jobs, superseded = merra2.plan(wanted, built, today, fetch)
    print(
        json.dumps([{"asset": j.asset_name, "year": j.year, "months": j.months[-1]} for j in jobs])
    )
    if args.superseded:
        Path(args.superseded).write_text("\n".join(superseded), encoding="utf-8")
    return 0


def _merra2_build(args: argparse.Namespace) -> int:
    job = merra2.Merra2Job(year=args.year, months=_months(args.months))
    print(merra2.build(job, Path(args.out), merra2.token_fetch(_earthdata_token())))
    return 0


def _earthdata_token() -> str:
    """NASA's Earthdata Login token, from `EARTHDATA_TOKEN` (the Actions secret of that name)."""
    token = os.environ.get("EARTHDATA_TOKEN", "")
    if not token:
        raise SystemExit("EARTHDATA_TOKEN is not set: NASA would refuse MERRA-2's files")
    return token


def _mirror(args: argparse.Namespace) -> int:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("HF_TOKEN is not set: Hugging Face would refuse the commits", file=sys.stderr)
        return 2
    release = mirror.load_release(Path(args.release))
    card = Path(args.card).read_bytes()
    with tempfile.TemporaryDirectory(dir=args.workdir) as workdir:
        done = mirror.sync(
            mirror.HubApi(token),
            release,
            card,
            Path(workdir),
            mirror.download_asset,
            folder=args.folder,
        )
    print(
        f"{len(release)} assets in the release: {len(done.upload)} uploaded, "
        f"{len(done.delete)} deleted, the card {'written' if done.card else 'unchanged'}"
    )
    return 0


def _stations(args: argparse.Namespace) -> int:
    last_year = args.last_year or date.today().year - 1
    pollutant = stations.Pollutant(args.pollutant)
    print(stations.build(args.country, pollutant, Path(args.out), last_year=last_year))
    return 0


def _static(args: argparse.Namespace) -> int:
    layers = Path(args.layers)
    path = static.build(
        layers / "population.tif",
        layers / "built_up.tif",
        layers / "elevation.tif",
        layers / "land.tif",
        Path(args.out),
        region=static.REGIONS[args.region],
    )
    print(path)
    return 0


def _fine(args: argparse.Namespace) -> int:
    layers = Path(args.layers)
    grid = Grid(Box(*args.box), fine.CELLS_PER_DEGREE)
    tiles = fine.build(layers, layers / "roads.geojsonseq", Path(args.out), grid)
    print(f"{len(tiles)} tiles in {args.out}")
    return 0


def _space(args: argparse.Namespace) -> int:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("HF_TOKEN is not set: Hugging Face would refuse the Space", file=sys.stderr)
        return 2
    out = Path(args.out)
    files = space.bundle(Path(args.source), Path(args.space), Path(args.requirements), out)
    space.deploy(HfApi(token=token), out)
    print(f"{len(files)} files of Panache deployed to {space.REPOSITORY}")
    return 0


def _fine_plan(args: argparse.Namespace) -> int:
    print(json.dumps(fine.plan(args.only)))
    return 0


def _months(text: str) -> tuple[int, ...]:
    """`all`, a month such as `2`, or a span such as `2-5`."""
    return ALL_MONTHS if text.strip() == "all" else plan.parse_span(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panache-archive",
        description="Daily air quality over Europe from the CAMS European reanalyses.",
    )
    commands = parser.add_subparsers(required=True)
    domains = [d.value for d in Domain]

    planning = commands.add_parser("plan", help="print the matrix of the assets still missing")
    planning.add_argument("--variables", default="all", help="`all`, or names such as pm10,ozone")
    planning.add_argument("--years", default="all", help="`all`, a year, or a span: 2013-2025")
    planning.add_argument("--built", help="a file listing the assets already built")
    planning.add_argument("--skip", default="", help="assets to leave out, separated by commas")
    planning.add_argument("--domain", default=Domain.FRANCE.value, choices=domains)
    planning.set_defaults(run=_plan)

    building = commands.add_parser("build", help="build one asset from the ADS")
    building.add_argument("--variable", required=True, choices=[v.value for v in Variable])
    building.add_argument("--year", required=True, type=int)
    building.add_argument(
        "--stream", default=Reanalysis.VALIDATED.value, choices=[s.value for s in Reanalysis]
    )
    building.add_argument("--months", default="all", help="`all`, a month, or a span: 2-5")
    building.add_argument("--domain", default=Domain.FRANCE.value, choices=domains)
    building.add_argument("--out", default="dist", help="where the asset is written")
    building.add_argument("--workdir", help="where downloads go (one month at a time)")
    building.set_defaults(run=_build)

    checking = commands.add_parser("check", help="compare assets with Open-Meteo")
    checking.add_argument("assets", nargs="+")
    checking.add_argument("--summary", help="a Markdown file to append the reports to")
    checking.add_argument(
        "--twins", help="a folder of France's assets, to compare the whole domain's with"
    )
    checking.set_defaults(run=_check)

    supervising = commands.add_parser("supervise", help="decide the next step, write the report")
    supervising.add_argument("--state", required=True, help="what the workflow gathered")
    supervising.add_argument("--out", required=True, help="where the decision is written")
    supervising.add_argument("--url", default="", help="this supervisor run's page")
    supervising.add_argument("--domain", default=Domain.FRANCE.value, choices=domains)
    supervising.set_defaults(run=_supervise)

    era5_planning = commands.add_parser("era5-plan", help="print the ERA5 years still missing")
    era5_planning.add_argument("--years", default="all", help="`all`, a year, or a span")
    era5_planning.add_argument("--built", help="a file listing the assets already built")
    era5_planning.add_argument("--domain", default=Domain.FRANCE.value, choices=domains)
    era5_planning.add_argument("--superseded", help="where to list the assets to delete")
    era5_planning.set_defaults(run=_era5_plan)

    era5_building = commands.add_parser("era5-build", help="build one ERA5 year from the bucket")
    era5_building.add_argument("--year", required=True, type=int)
    era5_building.add_argument("--months", default="all", help="`all`, a month, or a span: 2-5")
    era5_building.add_argument("--domain", default=Domain.FRANCE.value, choices=domains)
    era5_building.add_argument("--out", default="dist", help="where the asset is written")
    era5_building.set_defaults(run=_era5_build)

    merra2_planning = commands.add_parser(
        "merra2-plan", help="print the MERRA-2 assets still missing"
    )
    merra2_planning.add_argument("--years", default="all", help="`all`, a year, or a span")
    merra2_planning.add_argument("--built", help="a file listing the assets already built")
    merra2_planning.add_argument("--superseded", help="where to list the assets to delete")
    merra2_planning.set_defaults(run=_merra2_plan)

    merra2_building = commands.add_parser(
        "merra2-build", help="build one MERRA-2 asset from NASA's Cloud OPeNDAP"
    )
    merra2_building.add_argument("--year", required=True, type=int)
    merra2_building.add_argument("--months", default="all", help="`all`, a month, or a span: 1-9")
    merra2_building.add_argument("--out", default="dist", help="where the asset is written")
    merra2_building.set_defaults(run=_merra2_build)

    mirroring = commands.add_parser(
        "mirror", help="mirror the whole domain's release on Hugging Face"
    )
    mirroring.add_argument("--release", required=True, help="the release's JSON, from GitHub's API")
    mirroring.add_argument("--card", required=True, help="the dataset card to keep on the mirror")
    mirroring.add_argument("--folder", default=mirror.FOLDER, help="the release's folder there")
    mirroring.add_argument("--workdir", help="where each batch of assets is downloaded")
    mirroring.set_defaults(run=_mirror)

    stationing = commands.add_parser(
        "stations", help="build a country's daily means at its stations, from the EEA (E1a)"
    )
    stationing.add_argument("--country", default="FR", help="the EEA's two-letter code")
    stationing.add_argument(
        "--pollutant", required=True, choices=[p.value for p in stations.Pollutant]
    )
    stationing.add_argument("--last-year", type=int, help="the last year kept (default: last year)")
    stationing.add_argument("--out", default="dist", help="where the asset is written")
    stationing.set_defaults(run=_stations)

    fixing = commands.add_parser(
        "static", help="build the 1 km fixed layers from GDAL's cuts (scripts/static-layers.sh)"
    )
    fixing.add_argument("--layers", required=True, help="the folder of the cut GeoTIFFs")
    fixing.add_argument("--region", default="france", choices=sorted(static.REGIONS))
    fixing.add_argument("--out", default="dist", help="where the asset is written")
    fixing.set_defaults(run=_static)

    finer = commands.add_parser(
        "fine", help="build the 3 arc-second fixed layers' tiles (scripts/fine-layers.sh's cuts)"
    )
    finer.add_argument("--layers", required=True, help="the folder of the cuts and the roads")
    finer.add_argument("--out", default="dist", help="where the tiles are written")
    finer.add_argument(
        "--box",
        type=float,
        nargs=4,
        required=True,
        metavar=("SOUTH", "NORTH", "WEST", "EAST"),
        help="the block the cuts cover",
    )
    finer.set_defaults(run=_fine)
    spacing = commands.add_parser(
        "space", help="deploy the Hugging Face Space that computes views at ~100 m"
    )
    spacing.add_argument("--source", required=True, help="Panache's backend/src")
    spacing.add_argument("--space", required=True, help="the folder of its Dockerfile and card")
    spacing.add_argument("--requirements", required=True, help="the pinned dependencies")
    spacing.add_argument("--out", required=True, help="where the Space's files are gathered")
    spacing.set_defaults(run=_space)
    fine_plan = commands.add_parser(
        "fine-plan", help="print the fine layers' blocks as the workflow's matrix"
    )
    fine_plan.add_argument("--only", help="one block, such as N40E005")
    fine_plan.set_defaults(run=_fine_plan)
    return parser


if __name__ == "__main__":
    sys.exit(main())
