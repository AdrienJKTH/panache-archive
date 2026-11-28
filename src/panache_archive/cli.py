"""Command line: `panache-archive plan`, `build`, `check`, `supervise`, `era5-plan`,
`era5-build`, `merra2-plan`, `merra2-build`, `mirror`, `stations`, `live-stations`,
`live-compare`, `targets`, `static`, `private`,
`fine`, `fine-plan`, `space`, `winds-plan`, `winds-build`, `surface-plan`, `surface-build`,
`surface-pilot`, `emissions-plan`, `emissions-build`, `emissions-pilot`, `edgar-build` and
`countries`, as
the workflows run them."""

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
    countries,
    edgar,
    emissions,
    era5,
    fine,
    fine_present,
    live_compare,
    live_stations,
    merra2,
    mirror,
    move,
    plan,
    private,
    space,
    static,
    stations,
    supervise,
    surface,
    targets,
    winds,
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
    if private.is_private(args.folder):
        print(f"{args.folder} is private: it is not mirrored on the public dataset")
        return 0
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


def _private_store() -> private.PrivateStore:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise SystemExit("HF_TOKEN is not set: Hugging Face would refuse the private dataset")
    return private.PrivateStore(token)


def _present_plan(args: argparse.Namespace) -> int:
    names = Path(args.files).read_text(encoding="utf-8").splitlines()
    spans = fine_present.missing(names, date.fromisoformat(args.today))
    planned = [
        {"pollutant": s.pollutant, "first": s.first.isoformat(), "last": s.last.isoformat()}
        for s in spans
    ]
    print(json.dumps(planned))
    return 0


def _present_joins(args: argparse.Namespace) -> int:
    names = Path(args.files).read_text(encoding="utf-8").splitlines()
    found = fine_present.joins(names, date.fromisoformat(args.today))
    planned = [
        {
            "pollutant": j.whole.pollutant,
            "year": j.whole.first.year,
            "name": j.whole.name,
            "parts": [p.name for p in j.parts],
        }
        for j in found
    ]
    print(json.dumps(planned))
    return 0


def _present_forecasts(args: argparse.Namespace) -> int:
    names = Path(args.files).read_text(encoding="utf-8").splitlines()
    today = date.fromisoformat(args.today)
    planned = {
        "build": fine_present.without_forecast(names, today),
        "stale": fine_present.stale_forecasts(names, today),
    }
    print(json.dumps(planned))
    return 0


def _private_used(_args: argparse.Namespace) -> int:
    print(_private_store().used())
    return 0


def _private_list(args: argparse.Namespace) -> int:
    for name in _private_store().names(args.folder):
        print(name)
    return 0


def _private_upload(args: argparse.Namespace) -> int:
    files = [Path(name) for name in args.files]
    _private_store().upload(args.folder, files)
    print(f"{len(files)} files in the private dataset's {args.folder}")
    return 0


def _private_delete(args: argparse.Namespace) -> int:
    _private_store().delete(args.folder, args.names or None)
    print(f"{', '.join(args.names) if args.names else 'the whole folder'} out of {args.folder}")
    return 0


def _private_squash(_args: argparse.Namespace) -> int:
    _private_store().squash()
    print("the private dataset's history is squashed")
    return 0


def _private_import(args: argparse.Namespace) -> int:
    release = mirror.load_release(Path(args.release))
    with tempfile.TemporaryDirectory(dir=args.workdir) as workdir:
        copied = move.move(_private_store(), release, args.folder, Path(workdir))
    print(f"{len(release)} assets in the release: {len(copied)} copied into {args.folder}")
    return 0


def _private_download(args: argparse.Namespace) -> int:
    found = _private_store().download(args.folder, args.pattern, Path(args.dir))
    print(f"{len(found)} files from the private dataset's {args.folder} in {args.dir}")
    return 0 if found or args.allow_empty else 1


def _private_purge(args: argparse.Namespace) -> int:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise SystemExit("HF_TOKEN is not set: Hugging Face would refuse the commits")
    removed = private.purge_public(HfApi(token=token), mirror.REPO_ID, squash=args.squash)
    print(f"{len(removed)} private files removed from {mirror.REPO_ID}")
    for path in removed:
        print(path)
    print("history squashed" if args.squash else "history kept")
    return 0


def _stations(args: argparse.Namespace) -> int:
    last_year = args.last_year or date.today().year - 1
    pollutant = stations.Pollutant(args.pollutant)
    copy = Path(args.metadata_copy) if args.metadata_copy else None
    path = stations.build(
        args.country, pollutant, Path(args.out), last_year=last_year, metadata_copy=copy
    )
    print(path)
    return 0


def _live_stations(args: argparse.Namespace) -> int:
    yesterday = date.fromordinal(date.today().toordinal() - 1)
    first = date.fromisoformat(args.first_day) if args.first_day else yesterday
    last = date.fromisoformat(args.last_day) if args.last_day else first
    countries = args.countries.split(",") if args.countries else None
    pollutant = stations.Pollutant(args.pollutant)
    out = Path(args.out)
    copy = Path(args.metadata_copy) if args.metadata_copy else None
    for path in live_stations.build(
        pollutant, first, last, out, countries=countries, metadata_copy=copy
    ):
        print(path)
    return 0


def _live_compare(args: argparse.Namespace) -> int:
    locations = [int(x) for x in Path(args.locations).read_text().replace(",", " ").split()]
    folder = Path(args.validated)
    validated = {
        pollutant: live_compare.read_validated(
            sorted(folder.glob(f"stations-*-{pollutant}.parquet"))
        )
        for pollutant in live_stations.POLLUTANTS
        if any(folder.glob(f"stations-*-{pollutant}.parquet"))
    }
    result = live_compare.compare(locations, args.year, validated)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(live_compare.markdown(result))
    return 0


def _targets(args: argparse.Namespace) -> int:
    for path in targets.build(Path(args.out)):
        print(targets.describe(path), end="\n\n")
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


def _winds_plan(args: argparse.Namespace) -> int:
    built = set(Path(args.built).read_text(encoding="utf-8").split()) if args.built else set()
    years = plan.parse_years(args.years)
    wanted = sorted(years or range(winds.FIRST_YEAR, date.today().year + 1))
    print(json.dumps(winds.plan(wanted, built)))
    return 0


def _winds_build(args: argparse.Namespace) -> int:
    print(winds.build(args.year, args.month, Path(args.out), days=args.days))
    return 0


def _countries(args: argparse.Namespace) -> int:
    path = countries.build(Path(args.raster), Path(args.geojson), Path(args.out))
    print(path)
    for line in countries.describe(path):
        print(line)
    return 0


def _edgar_build(args: argparse.Namespace) -> int:
    years = sorted(plan.parse_years(args.years) or range(edgar.FIRST_YEAR, edgar.LAST_YEAR + 1))
    for path in edgar.build(years, Path(args.out), Path(args.workdir)):
        print(path)
    return 0


def _emissions_plan(args: argparse.Namespace) -> int:
    built = set(Path(args.built).read_text(encoding="utf-8").split()) if args.built else set()
    years = plan.parse_years(args.years)
    wanted = sorted(years or range(emissions.FIRST_YEAR, emissions.LAST_YEAR + 1))
    print(json.dumps(emissions.plan(wanted, built)))
    return 0


def _emissions_build(args: argparse.Namespace) -> int:
    print(emissions.build(args.species, args.year, Path(args.out), Path(args.workdir)))
    return 0


def _emissions_pilot(args: argparse.Namespace) -> int:
    print("\n".join(emissions.pilot(args.species, args.year, Path(args.workdir))))
    return 0


def _surface_plan(args: argparse.Namespace) -> int:
    built = set(Path(args.built).read_text(encoding="utf-8").split()) if args.built else set()
    years = plan.parse_years(args.years)
    wanted = sorted(years or range(surface.FIRST_YEAR, date.today().year + 1))
    print(json.dumps(surface.plan(wanted, built)))
    return 0


def _surface_build(args: argparse.Namespace) -> int:
    print(surface.build(args.year, args.month, Path(args.out), days=args.days))
    return 0


def _surface_pilot(args: argparse.Namespace) -> int:
    print("| Field | Unit kept | Min | Max | GB a year |")
    print("|---|---|---|---|---|")
    for row in surface.measure(args.year, args.month, args.days):
        print(
            f"| {row['field']} | {row['scale']:g} | {row['min']:.4g} | {row['max']:.4g} "
            f"| {row['gb_per_year']:.3f} |"
        )
    return 0


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

    winds_planning = commands.add_parser(
        "winds-plan", help="print the years whose ERA5 winds are out but not built"
    )
    winds_planning.add_argument("--years", default="all", help="`all`, a year, or a span")
    winds_planning.add_argument("--built", help="a file listing the assets already built")
    winds_planning.set_defaults(run=_winds_plan)

    winds_building = commands.add_parser(
        "winds-build", help="build one month of ERA5 winds from ARCO-ERA5"
    )
    winds_building.add_argument("--year", required=True, type=int)
    winds_building.add_argument("--month", required=True, type=int)
    winds_building.add_argument("--days", type=int, help="only the month's first days (pilots)")
    winds_building.add_argument("--out", default="dist", help="where the asset is written")
    winds_building.set_defaults(run=_winds_build)

    edgar_building = commands.add_parser(
        "edgar-build", help="build EDGAR's PM2.5 by sector over the winds' box, a file a year"
    )
    edgar_building.add_argument("--years", default="all", help="`all`, a year, or a span")
    edgar_building.add_argument("--out", default="dist", help="where the assets are written")
    edgar_building.add_argument("--workdir", default="work", help="where zips are downloaded")
    edgar_building.set_defaults(run=_edgar_build)

    bordering = commands.add_parser(
        "countries", help="build each emission cell's country (scripts/countries.sh's raster)"
    )
    bordering.add_argument("--raster", required=True, help="GDAL's raster of NE_IDs")
    bordering.add_argument("--geojson", required=True, help="Natural Earth's countries")
    bordering.add_argument("--out", default="dist", help="where the asset is written")
    bordering.set_defaults(run=_countries)

    emissions_planning = commands.add_parser(
        "emissions-plan", help="print the species and years of CAMS-GLOB-ANT not built yet"
    )
    emissions_planning.add_argument("--years", default="all", help="`all`, a year, or a span")
    emissions_planning.add_argument("--built", help="a file listing the assets already built")
    emissions_planning.set_defaults(run=_emissions_plan)

    for name, run, text in (
        ("emissions-build", _emissions_build, "build one species' year of CAMS-GLOB-ANT"),
        ("emissions-pilot", _emissions_pilot, "download a species' year and describe it"),
    ):
        emissions_command = commands.add_parser(name, help=text)
        emissions_command.add_argument("--species", required=True, choices=emissions.SPECIES)
        emissions_command.add_argument("--year", required=True, type=int)
        emissions_command.add_argument("--out", default="dist", help="where the asset is written")
        emissions_command.add_argument("--workdir", default="work", help="where files are fetched")
        emissions_command.set_defaults(run=run)

    surface_planning = commands.add_parser(
        "surface-plan", help="print the years whose ERA5 boundary layer is out but not built"
    )
    surface_planning.add_argument("--years", default="all", help="`all`, a year, or a span")
    surface_planning.add_argument("--built", help="a file listing the assets already built")
    surface_planning.set_defaults(run=_surface_plan)

    for name, run, text in (
        ("surface-build", _surface_build, "build one month of ERA5's boundary layer"),
        ("surface-pilot", _surface_pilot, "measure what each candidate field would weigh"),
    ):
        surface_command = commands.add_parser(name, help=text)
        surface_command.add_argument("--year", required=True, type=int)
        surface_command.add_argument("--month", required=True, type=int)
        surface_command.add_argument("--days", type=int, help="only the month's first days")
        surface_command.add_argument("--out", default="dist", help="where the asset is written")
        surface_command.set_defaults(run=run)

    mirroring = commands.add_parser(
        "mirror", help="mirror the whole domain's release on Hugging Face"
    )
    mirroring.add_argument("--release", required=True, help="the release's JSON, from GitHub's API")
    mirroring.add_argument("--card", required=True, help="the dataset card to keep on the mirror")
    mirroring.add_argument("--folder", default=mirror.FOLDER, help="the release's folder there")
    mirroring.add_argument("--workdir", help="where each batch of assets is downloaded")
    mirroring.set_defaults(run=_mirror)

    privately = commands.add_parser(
        "private", help="the private Hugging Face dataset of models and experiment results"
    ).add_subparsers(dest="action", required=True)
    for name, run, said in (
        ("present-plan", _present_plan, "the days the present's 1 km maps lack, as JSON"),
        ("present-joins", _present_joins, "the months of those maps to join, as JSON"),
        ("present-forecasts", _present_forecasts, "the forecasts to make and to drop, as JSON"),
    ):
        planning = commands.add_parser(name, help=said)
        planning.add_argument("--files", required=True, help="the folder's names, one a line")
        planning.add_argument("--today", required=True, help="the day it is, ISO")
        planning.set_defaults(run=run)
    privately.add_parser(
        "used", help="the bytes the private dataset takes of the quota, history included"
    ).set_defaults(run=_private_used)
    private_list = privately.add_parser("list", help="the files of a folder, one a line")
    private_list.add_argument("--folder", required=True, help="a release's name")
    private_list.set_defaults(run=_private_list)
    private_upload = privately.add_parser(
        "upload", help="files (or directories, with the files under them) into a folder, one commit"
    )
    private_upload.add_argument("--folder", required=True, help="a release's name")
    private_upload.add_argument("files", nargs="+", help="the files or directories to upload")
    private_upload.set_defaults(run=_private_upload)
    private_delete = privately.add_parser(
        "delete", help="files out of a folder, or the whole folder without names"
    )
    private_delete.add_argument("--folder", required=True, help="a release's name")
    private_delete.add_argument("names", nargs="*", help="the files to delete")
    private_delete.set_defaults(run=_private_delete)
    private_squash = privately.add_parser(
        "squash", help="squash the dataset's history (irreversible)"
    )
    private_squash.set_defaults(run=_private_squash)
    private_import = privately.add_parser(
        "import", help="copy a public release into a folder, a batch at a time"
    )
    private_import.add_argument("--release", required=True, help="the release's JSON, from GitHub")
    private_import.add_argument("--folder", required=True, help="its name there")
    private_import.add_argument("--workdir", help="where each batch of assets is downloaded")
    private_import.set_defaults(run=_private_import)
    private_download = privately.add_parser(
        "download", help="the files of a folder that match a pattern; fails when none does"
    )
    private_download.add_argument("--folder", required=True, help="a release's name")
    private_download.add_argument("--pattern", default="*", help="a glob over the file names")
    private_download.add_argument("--dir", required=True, help="where the files are written")
    private_download.add_argument(
        "--allow-empty", action="store_true", help="succeed when no file matches"
    )
    private_download.set_defaults(run=_private_download)

    private_purge = privately.add_parser(
        "purge-public", help="remove from the public dataset what must never be there"
    )
    private_purge.add_argument(
        "--squash", action="store_true", help="then squash its history (irreversible)"
    )
    private_purge.set_defaults(run=_private_purge)

    stationing = commands.add_parser(
        "stations", help="build a country's daily means at its stations, from the EEA (E1a)"
    )
    stationing.add_argument("--country", default="FR", help="the EEA's two-letter code")
    stationing.add_argument(
        "--pollutant", required=True, choices=[p.value for p in stations.Pollutant]
    )
    stationing.add_argument("--last-year", type=int, help="the last year kept (default: last year)")
    stationing.add_argument("--out", default="dist", help="where the asset is written")
    stationing.add_argument(
        "--metadata-copy", help="the EEA's station metadata kept, read when its server fails"
    )
    stationing.set_defaults(run=_stations)

    living = commands.add_parser(
        "live-stations", help="build days of daily values at Europe's stations, from the EEA (E2a)"
    )
    living.add_argument(
        "--pollutant", required=True, choices=[p.value for p in live_stations.BUILDABLE]
    )
    living.add_argument("--first-day", help="the first UTC day, ISO (default: yesterday)")
    living.add_argument("--last-day", help="the last UTC day, ISO (default: the first)")
    living.add_argument("--countries", help="two-letter codes, e.g. FR,DE (default: every one)")
    living.add_argument("--out", default="dist", help="where the assets are written")
    living.add_argument(
        "--metadata-copy",
        help="a gzipped copy of the EEA's station metadata: written when the EEA answers, "
        "read when it does not",
    )
    living.set_defaults(run=_live_stations)

    comparing = commands.add_parser(
        "live-compare", help="score OpenAQ's archive of the EEA's live hours against validated days"
    )
    comparing.add_argument("--year", type=int, required=True, help="the year compared")
    comparing.add_argument("--locations", required=True, help="a file of OpenAQ location ids")
    comparing.add_argument("--validated", required=True, help="a folder of validated assets")
    comparing.add_argument("--out", default="compare.json", help="the scores, as JSON")
    comparing.set_defaults(run=_live_compare)

    targeting = commands.add_parser(
        "targets", help="build the measured sources of carbonaceous aerosol, from Zenodo"
    )
    targeting.add_argument("--out", default="dist", help="where the assets are written")
    targeting.set_defaults(run=_targets)

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
