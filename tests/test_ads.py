import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from panache_archive import ads
from panache_archive.variables import Reanalysis, Variable


def test_a_request_asks_for_one_variable_month_at_the_surface() -> None:
    assert ads.request(Variable.PM2_5, 2021, 2, Reanalysis.VALIDATED) == {
        "variable": ["particulate_matter_2.5um"],
        "model": ["ensemble"],
        "level": ["0"],
        "type": ["validated_reanalysis"],
        "year": ["2021"],
        "month": ["02"],
    }


class Constraints:
    """The ADS constraints endpoint, offering `years`, and `months` in its latest year."""

    def __init__(self, years: list[str], months: list[str]) -> None:
        self.years, self.months = years, months
        self.asked: list[dict[str, Any]] = []

    def __call__(self, url: str, body: bytes) -> bytes:
        assert url == ads.CONSTRAINTS_URL
        inputs = json.loads(body)["inputs"]
        self.asked.append(inputs)
        if "year" in inputs:
            return json.dumps({"year": inputs["year"], "month": self.months}).encode()
        return json.dumps(
            {"year": self.years, "month": [f"{m:02d}" for m in range(1, 13)]}
        ).encode()


def test_complete_years_are_every_offered_year_when_the_latest_is_whole() -> None:
    post = Constraints(["2023", "2024", "2025"], [f"{m:02d}" for m in range(1, 13)])

    years = ads.complete_years(Variable.DUST, Reanalysis.INTERIM, post)

    assert years == {2023, 2024, 2025}
    assert post.asked[0] == {
        "variable": ["dust"],
        "model": ["ensemble"],
        "level": ["0"],
        "type": ["interim_reanalysis"],
    }
    assert post.asked[1]["year"] == ["2025"]


def test_a_latest_year_still_growing_is_left_out() -> None:
    post = Constraints(["2025", "2026"], ["01", "02", "03"])

    assert ads.complete_years(Variable.OZONE, Reanalysis.INTERIM, post) == {2025}


def test_no_offered_year_needs_no_second_question() -> None:
    post = Constraints([], [])

    assert ads.complete_years(Variable.PM10, Reanalysis.VALIDATED, post) == frozenset()
    assert len(post.asked) == 1


def test_a_zip_gives_its_netcdf_files_and_is_removed(tmp_path: Path) -> None:
    download = tmp_path / "pm10.download"
    with zipfile.ZipFile(download, "w") as archive:
        archive.writestr("b.nc", b"second")
        archive.writestr("a.nc", b"first")
        archive.writestr("README.txt", b"not netCDF")
    workdir = tmp_path / "work"
    workdir.mkdir()

    files = ads.netcdf_files(download, workdir)

    assert [f.name for f in files] == ["a.nc", "b.nc"]
    assert files[0].read_bytes() == b"first"
    assert not download.exists()


def test_a_plain_download_is_the_netcdf_file(tmp_path: Path) -> None:
    download = tmp_path / "pm10.download"
    download.write_bytes(b"CDF")

    assert ads.netcdf_files(download, tmp_path) == [tmp_path / "pm10.nc"]


def test_a_zip_without_netcdf_is_an_error(tmp_path: Path) -> None:
    download = tmp_path / "pm10.download"
    with zipfile.ZipFile(download, "w") as archive:
        archive.writestr("README.txt", b"nothing")

    with pytest.raises(ValueError, match="no netCDF"):
        ads.netcdf_files(download, tmp_path)


def test_download_goes_through_cdsapi(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []

    class Client:
        def __init__(self, **options: Any) -> None:
            calls.append(("client", options))

        def retrieve(self, dataset: str, request: dict[str, list[str]], target: str) -> None:
            calls.append(("retrieve", (dataset, request, target)))

    monkeypatch.setattr("cdsapi.Client", Client)
    request = ads.request(Variable.PM10, 2021, 2, Reanalysis.VALIDATED)

    ads.download(request, tmp_path / "x.download")

    assert calls == [
        ("client", {"quiet": True, "progress": False}),
        ("retrieve", (ads.DATASET, dict(request), str(tmp_path / "x.download"))),
    ]


def test_post_json_sends_the_body(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[Any] = []

    def urlopen(request: Any, timeout: float) -> io.BytesIO:
        sent.append((request.full_url, request.data, request.get_header("Content-type"), timeout))
        return io.BytesIO(b'{"year": []}')

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    assert ads.post_json("https://example.com/c", b"{}") == b'{"year": []}'
    assert sent == [("https://example.com/c", b"{}", "application/json", ads.TIMEOUT_SECONDS)]
