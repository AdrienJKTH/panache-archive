import io
import zipfile
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from panache_archive import cli, edgar, emissions

BOX = emissions.Box(south=40.5, west=-9.5, step=1.0, rows=10, columns=20, tile=8)
LATS = np.arange(-89.5, 90.0, 1.0)
LONS = np.arange(-179.5, 180.0, 1.0)
FOLDERS = (*edgar.SURFACE, *edgar.ALOFT)
YEARS = (2021, 2022)


def flux(folder: str, year: int) -> float:
    """What a sector emits in every cell of a year, kg m⁻² s⁻¹; the residential sector one cell
    only, and far more."""
    return (1 + FOLDERS.index(folder)) * 1e-12 * (1 + (year - 2021) * 0.5)


def grid(folder: str, year: int, total_factor: float, uncounted: str = "") -> np.ndarray:
    """A sector's map, or the total of every sector but the `uncounted` one."""
    values = np.zeros((LATS.size, LONS.size), dtype=np.float32)
    if folder == edgar.TOTAL:
        for sector in FOLDERS:
            if sector != uncounted:
                values += grid(sector, year, 1.0)
        return values * total_factor
    if folder == "RCO":
        values[135, 185] = 2e-7
        values[0, 0] = np.nan
        return values
    values[:] = flux(folder, year)
    return values


def downloader(
    total_factor: float = 1.0,
    years: tuple[int, ...] = YEARS,
    absent: str = "",
    counted: bool = False,
) -> edgar.Download:
    """Zips of `years` for every sector; the `absent` one has no file at all, and the total
    counts it only if `counted`."""
    asked: list[str] = []

    def download(url: str, target: Path) -> None:
        asked.append(url)
        folder = url.split("/")[-2]
        assert url == f"{edgar.BASE}/PM2.5/{folder}/{folder}_flx_nc.zip"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("_readme.html", "x")
            for year in years if folder != absent else ():
                name = f"v8.1_FT2022_AP_PM2.5_{year}_{folder}_flx.nc"
                dataset = netCDF4.Dataset(name, "w", memory=1024)
                dataset.createDimension("lat", LATS.size)
                dataset.createDimension("lon", LONS.size)
                dataset.createVariable("lat", "f8", ("lat",))[:] = LATS
                dataset.createVariable("lon", "f8", ("lon",))[:] = LONS
                fluxes = dataset.createVariable("fluxes", "f4", ("lat", "lon"), fill_value=np.nan)
                fluxes[:] = grid(folder, year, total_factor, "" if counted else absent)
                archive.writestr(name, bytes(dataset.close()))

    download.asked = asked  # type: ignore[attr-defined]
    return download


def test_years_are_built_and_read_back(tmp_path: Path) -> None:
    download = downloader()
    paths = edgar.build(YEARS, tmp_path / "dist", tmp_path / "work", download, box=BOX)
    assert [path.name for path in paths] == [
        "emissions-edgar_pm25-2021.emissions",
        "emissions-edgar_pm25-2022.emissions",
    ]
    assert len(download.asked) == 24  # type: ignore[attr-defined]
    assert list((tmp_path / "work").iterdir()) == []
    for path, year in zip(paths, YEARS, strict=True):
        data = path.read_bytes()
        header = emissions.Header.unpack(data)
        assert (header.year, header.periods, header.scale) == (year, 1, edgar.SCALE)
        assert header.sectors == tuple(edgar.SURFACE.values())
        assert all(len(name) <= 8 for name in header.sectors)
        whole = np.concatenate(
            [
                np.concatenate(
                    [emissions.read_tile(data, 0, row, column) for column in range(3)], axis=-1
                )
                for row in range(2)
            ],
            axis=-2,
        )
        assert whole.shape == (20, 10, 20)
        for index, folder in enumerate(edgar.SURFACE):
            if folder == "RCO":
                # One cell at 45.5° N, 5.5° E, above what CAMS-GLOB-ANT's scale would hold.
                assert whole[index, 5, 15] == pytest.approx(2e-7, rel=1e-3)
                assert np.count_nonzero(whole[index]) == 1
            else:
                np.testing.assert_allclose(whole[index], flux(folder, year), rtol=1e-3)


def test_sectors_that_do_not_add_up_to_the_total_stop_the_build(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="add up"):
        edgar.build([2021], tmp_path / "dist", tmp_path / "work", downloader(1.2), box=BOX)
    assert list((tmp_path / "work").iterdir()) == []


def test_a_sector_without_a_file_for_the_year_emitted_nothing(tmp_path: Path) -> None:
    """EDGAR's supersonic sector has no file after 2003, and its total counts nothing for it."""
    download = downloader(absent="TNR_Aviation_SPS")
    [path] = edgar.build([2021], tmp_path / "dist", tmp_path / "work", download, box=BOX)
    assert emissions.Header.unpack(path.read_bytes()).sectors == tuple(edgar.SURFACE.values())

    # A surface sector with no file is written as zeros.
    download = downloader(absent="AGS")
    [path] = edgar.build([2021], tmp_path / "dist", tmp_path / "work", download, box=BOX)
    assert not emissions.read_tile(path.read_bytes(), 0, 0, 0)[0].any()
    assert emissions.read_tile(path.read_bytes(), 0, 0, 0)[1].all()


def test_a_missing_file_the_total_counts_is_caught(tmp_path: Path) -> None:
    download = downloader(absent="TRO", counted=True)
    with pytest.raises(ValueError, match="add up"):
        edgar.build([2021], tmp_path / "dist", tmp_path / "work", download, box=BOX)


def test_a_total_or_a_year_stored_twice_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="total has no file for 2021"):
        edgar.build(
            [2021], tmp_path / "dist", tmp_path / "work", downloader(absent="TOTALS"), box=BOX
        )
    with zipfile.ZipFile(tmp_path / "twice.zip", "w") as archive:
        archive.writestr("a_2021_x.nc", b"")
        archive.writestr("b_2021_x.nc", b"")
    with (
        zipfile.ZipFile(tmp_path / "twice.zip") as archive,
        pytest.raises(ValueError, match="found 2"),
    ):
        edgar.fluxes(archive, 2021, BOX)


def test_a_download_is_streamed_and_tried_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def urlopen(_request: object, timeout: float) -> io.BytesIO:
        calls.append(timeout)
        if len(calls) == 1:
            raise ConnectionResetError("cut")
        return io.BytesIO(b"zip bytes")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    edgar.fetch("https://jrc.test/x.zip", tmp_path / "x.zip")
    assert (tmp_path / "x.zip").read_bytes() == b"zip bytes"
    assert len(calls) == 2

    def always_cut(_request: object, timeout: float) -> io.BytesIO:  # noqa: ARG001
        raise ConnectionResetError("cut")

    monkeypatch.setattr("urllib.request.urlopen", always_cut)
    with pytest.raises(ConnectionResetError):
        edgar.fetch("https://jrc.test/x.zip", tmp_path / "y.zip")


def test_a_year_of_the_wrong_shape_is_refused(tmp_path: Path) -> None:
    header = emissions.Header(2021, ("RCO",), BOX, periods=1, scale=edgar.SCALE)
    with pytest.raises(ValueError, match="1 periods of 1 sectors"):
        emissions.write(tmp_path / "x", header, np.zeros((12, 1, 10, 20), dtype=np.float32))


def test_the_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_build(years: list[int], out: Path, workdir: Path) -> list[Path]:
        return [out / f"{year}-{workdir.name}" for year in years]

    monkeypatch.setattr(edgar, "build", fake_build)
    arguments = ["--out", str(tmp_path), "--workdir", "w"]
    assert cli.main(["edgar-build", "--years", "2020-2021", *arguments]) == 0
    assert [Path(line).name for line in capsys.readouterr().out.split()] == ["2020-w", "2021-w"]
    assert cli.main(["edgar-build", *arguments]) == 0
    assert len(capsys.readouterr().out.split()) == 10
