import json

import pytest

from panache_archive import plan
from panache_archive.build import Job
from panache_archive.variables import Reanalysis, Variable

OFFERED = {
    (Variable.PM10, Reanalysis.VALIDATED): frozenset({2023, 2024}),
    (Variable.PM10, Reanalysis.INTERIM): frozenset({2024, 2025}),
    (Variable.DUST, Reanalysis.VALIDATED): frozenset({2024}),
    (Variable.DUST, Reanalysis.INTERIM): frozenset(),
}


def offered(variable: Variable, stream: Reanalysis) -> frozenset[int]:
    return OFFERED[(variable, stream)]


def test_the_validated_stream_comes_first_and_the_interim_one_fills_in() -> None:
    jobs = plan.wanted([Variable.PM10, Variable.DUST], None, offered)

    assert jobs == [
        Job(Variable.PM10, 2023, Reanalysis.VALIDATED),
        Job(Variable.PM10, 2024, Reanalysis.VALIDATED),
        Job(Variable.PM10, 2025, Reanalysis.INTERIM),
        Job(Variable.DUST, 2024, Reanalysis.VALIDATED),
    ]


def test_years_narrow_the_jobs() -> None:
    jobs = plan.wanted([Variable.PM10], frozenset({2025, 2030}), offered)

    assert jobs == [Job(Variable.PM10, 2025, Reanalysis.INTERIM)]


def test_built_assets_are_not_built_again() -> None:
    jobs = plan.wanted([Variable.PM10], None, offered)

    left = plan.missing(
        jobs, {"cams-pm10-2023-validated.parquet", "cams-pm10-2025-validated.parquet"}
    )

    assert [job.year for job in left] == [2024, 2025]


def test_the_matrix_lists_each_job() -> None:
    matrix = plan.matrix([Job(Variable.OZONE, 2021, Reanalysis.VALIDATED)])

    assert json.loads(matrix) == [
        {"variable": "ozone", "year": 2021, "stream": "validated_reanalysis"}
    ]
    assert plan.matrix([]) == "[]"


def test_variables_are_all_or_named() -> None:
    assert plan.parse_variables("all") == list(Variable)
    assert plan.parse_variables(" pm10, ozone ,") == [Variable.PM10, Variable.OZONE]
    with pytest.raises(ValueError, match="pm1"):
        plan.parse_variables("pm1")


def test_years_are_all_one_or_a_span() -> None:
    assert plan.parse_years("all") is None
    assert plan.parse_years("2021") == frozenset({2021})
    assert plan.parse_years("2013-2015") == frozenset({2013, 2014, 2015})
    with pytest.raises(ValueError, match="backwards"):
        plan.parse_years("2015-2013")
