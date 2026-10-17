"""The Atmosphere Data Store (ADS): what the CAMS European reanalyses offer, and their files.

Dataset `cams-europe-air-quality-reanalyses` (DOI 10.24381/7cc0465a, CC-BY 4.0). It cannot
be cut to an area: every request returns the whole European grid, hence this pipeline runs on
GitHub's machines and keeps France only. A request costs one unit per variable-month, at most
10; one month at a time keeps the disk to one file (~0.8 GB).
"""

import json
import shutil
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, Protocol

import cdsapi

from panache_archive import web
from panache_archive.variables import ADS_NAME, Reanalysis, Variable

DATASET: Final = "cams-europe-air-quality-reanalyses"
DOI: Final = "10.24381/7cc0465a"
API_URL: Final = "https://ads.atmosphere.copernicus.eu/api"
CONSTRAINTS_URL: Final = f"{API_URL}/retrieve/v1/processes/{DATASET}/constraints"
MONTHS: Final = 12

Request = Mapping[str, list[str]]
Post = Callable[[str, bytes], bytes]
"""Sends a JSON body to a URL and returns the answer's body."""


class Download(Protocol):
    """Fetches the answer to an ADS request into `target`."""

    def __call__(self, request: Request, target: Path, /) -> None: ...


def request(variable: Variable, year: int, month: int, stream: Reanalysis) -> Request:
    """The ADS request for one variable over one month."""
    return {**_selection(variable, stream), "year": [str(year)], "month": [f"{month:02d}"]}


def download(request: Request, target: Path) -> None:
    """Fetch with the ADS key of `CDSAPI_URL` and `CDSAPI_KEY`, or of `~/.cdsapirc`."""
    client = cdsapi.Client(quiet=True, progress=False)
    client.retrieve(DATASET, dict(request), str(target))


def post_json(url: str, body: bytes) -> bytes:
    """POST a JSON body (the constraints endpoint needs no key)."""
    sent = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    return web.read(sent)


def complete_years(
    variable: Variable, stream: Reanalysis, post: Post = post_json
) -> frozenset[int]:
    """The years the ADS offers whole, all 12 months, for `variable` in `stream`.

    Only the latest year can be partial: the streams grow month by month.
    """
    years = frozenset(int(y) for y in _constraints(variable, stream, post, year=None)["year"])
    if not years:
        return years
    latest = max(years)
    months = _constraints(variable, stream, post, year=latest)["month"]
    return years if len(months) == MONTHS else years - {latest}


def netcdf_files(download: Path, workdir: Path) -> list[Path]:
    """The netCDF files of a download, which is a zip of them or a single one; it is removed."""
    if not zipfile.is_zipfile(download):
        return [download.rename(download.with_suffix(".nc"))]
    files: list[Path] = []
    with zipfile.ZipFile(download) as archive:
        for member in archive.infolist():
            if not member.filename.endswith(".nc"):
                continue
            path = workdir / Path(member.filename).name
            with archive.open(member) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target)
            files.append(path)
    download.unlink()
    if not files:
        raise ValueError(f"{download.name} holds no netCDF file")
    return sorted(files)


def _selection(variable: Variable, stream: Reanalysis) -> dict[str, list[str]]:
    """One variable at the surface, from the ensemble median of the models."""
    return {
        "variable": [ADS_NAME[variable]],
        "model": ["ensemble"],
        "level": ["0"],
        "type": [stream.value],
    }


def _constraints(
    variable: Variable, stream: Reanalysis, post: Post, *, year: int | None
) -> dict[str, list[str]]:
    """The years, or a year's months, that the ADS offers for the selection."""
    inputs = _selection(variable, stream)
    if year is not None:
        inputs["year"] = [str(year)]
    answer = json.loads(post(CONSTRAINTS_URL, json.dumps({"inputs": inputs}).encode()))
    return {"year": list(answer.get("year", [])), "month": list(answer.get("month", []))}
