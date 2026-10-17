"""Which assets the archive still lacks: the work of a run, as a GitHub Actions matrix.

Each variable-year comes from the validated stream when the ADS offers it whole, else from the
interim one. A year built from the interim stream is built again once the validated one covers
it, under another name; the interim asset can then be deleted.
"""

import json
from collections.abc import Callable, Collection, Iterable

from panache_archive.build import Job
from panache_archive.variables import Reanalysis, Variable

Years = Callable[[Variable, Reanalysis], frozenset[int]]
"""The years a stream offers whole for a variable (`ads.complete_years`)."""


def wanted(
    variables: Iterable[Variable], years: Collection[int] | None, offered: Years
) -> list[Job]:
    """Every whole-year job of `variables` in `years` (all years if `None`) that the ADS offers."""
    jobs: list[Job] = []
    for variable in variables:
        validated = offered(variable, Reanalysis.VALIDATED)
        interim = offered(variable, Reanalysis.INTERIM)
        for year in sorted(validated | interim):
            if years is not None and year not in years:
                continue
            stream = Reanalysis.VALIDATED if year in validated else Reanalysis.INTERIM
            jobs.append(Job(variable=variable, year=year, stream=stream))
    return jobs


def missing(jobs: Iterable[Job], built: Collection[str]) -> list[Job]:
    """The jobs whose asset is not among the `built` asset names."""
    return [job for job in jobs if job.asset_name not in built]


def matrix(jobs: Iterable[Job]) -> str:
    """The jobs as the JSON list of a matrix's `include`; `asset` names each job."""
    return json.dumps(
        [
            {
                "asset": j.asset_name,
                "variable": j.variable.value,
                "year": j.year,
                "stream": j.stream.value,
            }
            for j in jobs
        ]
    )


def parse_variables(text: str) -> list[Variable]:
    """`all`, or Open-Meteo names separated by commas, such as `pm10,ozone`."""
    if text.strip() == "all":
        return list(Variable)
    return [Variable(name.strip()) for name in text.split(",") if name.strip()]


def parse_years(text: str) -> frozenset[int] | None:
    """`all` (`None`), a year such as `2021`, or a span such as `2013-2025`."""
    return None if text.strip() == "all" else frozenset(parse_span(text))


def parse_span(text: str) -> tuple[int, ...]:
    """A number such as `2`, or a span such as `2-5`, as the numbers it covers."""
    first, _, last = text.strip().partition("-")
    start, end = int(first), int(last or first)
    if start > end:
        raise ValueError(f"{text!r} runs backwards")
    return tuple(range(start, end + 1))
