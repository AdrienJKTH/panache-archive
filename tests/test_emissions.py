import json
import zipfile
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from panache_archive import cli, emissions

BOX = emissions.Box(south=40.5, west=-9.5, step=1.0, rows=10, columns=20, tile=8)
LATS = np.arange(-89.5, 90.0, 1.0)
LONS = np.arange(-179.5, 180.0, 1.0)


def inventory(
    path: Path, lat_name: str = "lat", lon_name: str = "lon", total_factor: float = 1.0
) -> Path:
    """A year of two sectors at 1°: `res` emits 2e-12 kg/m²/s over land east of 0°, `ene` in
    one cell."""
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 12)
        dataset.createDimension(lat_name, LATS.size)
        dataset.createDimension(lon_name, LONS.size)
        dataset.createVariable("time", "f8", ("time",))[:] = np.arange(12) * 30.0
        dataset.variables["time"].units = "days since 2021-01-01"
        dataset.createVariable(lat_name, "f8", (lat_name,))[:] = LATS
        dataset.createVariable(lon_name, "f8", (lon_name,))[:] = LONS
        res = dataset.createVariable("res", "f4", ("time", lat_name, lon_name))
        res.units, res.long_name = "kg m-2 s-1", "Residential"
        values = np.zeros((12, LATS.size, LONS.size), dtype=np.float32)
        values[:, :, LONS > 0] = 2e-12
        res[:] = values
        ene = dataset.createVariable("ene", "f4", ("time", lat_name, lon_name))
        ene.units = "kg m-2 s-1"
        point = np.zeros_like(values)
        point[:, 135, 185] = 1e-9
        ene[:] = point
        total = dataset.createVariable("sum", "f4", ("time", lat_name, lon_name))
        total.units = "kg m-2 s-1"
        total[:] = (values + point) * total_factor
        dataset.createVariable("gridcell_area", "f4", (lat_name, lon_name))[:] = 1.0
    return path


def test_a_file_is_described_over_the_box(tmp_path: Path) -> None:
    lines = emissions.describe(inventory(tmp_path / "bc.nc"), BOX)
    assert lines[0].startswith("**bc.nc**")
    assert "time 12" in lines[1]
    assert "- lat: -89.5 to 89.5" in lines[2]
    fields = {
        line.split(" | ")[0].strip("| "): line.split(" | ") for line in lines if "kg m-2" in line
    }
    assert set(fields) == {"res", "ene", "sum"}
    # Half the box lies east of 0°: 100 cells of 2e-12 kg/m²/s.
    areas = emissions._areas(BOX.lats, 1.0)
    expected = 2e-12 * areas.sum() * 10 * emissions.SECONDS_PER_YEAR / 1e6
    assert float(fields["res"][3]) == pytest.approx(expected, rel=0.01)
    assert fields["res"][1] == "Residential"
    assert fields["res"][4] == "50.0%"
    assert fields["ene"][4] == "0.5%"
    assert len(emissions.encoded(np.ones((12, 10, 20), dtype=np.float32))) > 0


def test_other_axis_names_and_a_grid_without_the_cells(tmp_path: Path) -> None:
    path = inventory(tmp_path / "bc.nc", "latitude", "longitude")
    assert any(line.startswith("| res ") for line in emissions.describe(path, BOX))
    shifted = emissions.Box(south=40.25, west=-9.5, step=1.0, rows=2, columns=2, tile=2)
    with pytest.raises(ValueError, match="does not hold"):
        emissions.describe(path, shifted)


@pytest.mark.parametrize("zipped", [True, False])
def test_the_pilot_downloads_and_describes(tmp_path: Path, zipped: bool) -> None:
    asked = []

    def fetch(wanted: dict[str, list[str]], target: Path) -> None:
        asked.append(wanted)
        source = inventory(tmp_path / "CAMS-GLOB-ANT_bc.nc")
        if zipped:
            with zipfile.ZipFile(target, "w") as archive:
                archive.write(source, source.name)
            source.unlink()
        else:
            source.rename(target)

    lines = emissions.pilot("black_carbon", 2021, tmp_path / "work", fetch, BOX)
    assert asked == [
        {
            "variable": ["black_carbon"],
            "source": ["anthropogenic"],
            "version": [emissions.VERSION],
            "year": ["2021"],
        }
    ]
    assert lines[0].startswith("### black_carbon, 2021")
    assert any(line.startswith("| ene ") for line in lines)
    assert list((tmp_path / "work").iterdir()) == []


def test_the_download_asks_the_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[object] = []

    class Client:
        def __init__(self, **options: bool) -> None:
            calls.append(options)

        def retrieve(self, dataset: str, wanted: dict[str, list[str]], target: str) -> None:
            calls.append((dataset, wanted, target))

    monkeypatch.setattr("cdsapi.Client", Client)
    wanted = emissions.request("organic_carbon", 2020)
    emissions.download(wanted, tmp_path / "oc")
    assert calls[1] == (emissions.DATASET, wanted, str(tmp_path / "oc"))


def test_the_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_pilot(species: str, year: int, workdir: Path) -> list[str]:
        return [f"{species} {year} {workdir.name}"]

    monkeypatch.setattr(emissions, "pilot", fake_pilot)
    arguments = ["--species", "black_carbon", "--year", "2021", "--workdir", str(tmp_path / "w")]
    assert cli.main(["emissions-pilot", *arguments]) == 0
    assert capsys.readouterr().out.strip() == "black_carbon 2021 w"


def fetcher(tmp_path: Path, total_factor: float = 1.0) -> emissions.Download:
    def fetch(_wanted: dict[str, list[str]], target: Path) -> None:
        inventory(tmp_path / "source.nc", total_factor=total_factor).rename(target)

    return fetch


def test_a_year_is_built_and_read_back_tile_by_tile(tmp_path: Path) -> None:
    path = emissions.build(
        "black_carbon", 2021, tmp_path / "dist", tmp_path / "work", fetcher(tmp_path), box=BOX
    )
    assert path.name == "emissions-black_carbon-2021.emissions"
    assert list((tmp_path / "work").iterdir()) == []
    data = path.read_bytes()
    header = emissions.Header.unpack(data)
    assert (header.year, header.sectors, header.box) == (2021, ("res", "ene"), BOX)
    for month in (0, 11):
        whole = np.concatenate(
            [
                np.concatenate(
                    [emissions.read_tile(data, month, row, column) for column in range(3)], axis=-1
                )
                for row in range(2)
            ],
            axis=-2,
        )
        assert whole.shape == (2, 10, 20)
        # East of 0° the residential sector emits 2e-12 kg/m²/s; float16 keeps three digits.
        np.testing.assert_allclose(whole[0, :, BOX.lons > 0], 2e-12, rtol=1e-3)
        assert not whole[0, :, BOX.lons < 0].any()
        # The power plant sits at 45.5° N, 5.5° E.
        assert whole[1, 5, 15] == pytest.approx(1e-9, rel=1e-3)
        assert np.count_nonzero(whole[1]) == 1


def test_sectors_that_do_not_add_up_stop_the_build(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="add up"):
        emissions.build(
            "black_carbon",
            2021,
            tmp_path / "dist",
            tmp_path / "work",
            fetcher(tmp_path, total_factor=1.5),
            box=BOX,
        )
    assert list((tmp_path / "work").iterdir()) == []


def test_values_beyond_float16_stop_the_write(tmp_path: Path) -> None:
    header = emissions.Header(2021, ("res",), BOX)
    for value in (1e-6, -1e-12, np.nan):
        year = np.full((12, 1, 10, 20), value, dtype=np.float32)
        with pytest.raises(ValueError, match="beyond float16"):
            emissions.write(tmp_path / "x.emissions", header, year)


def test_the_header_and_frames_are_checked(tmp_path: Path) -> None:
    header = emissions.Header(2021, ("res", "tro", "shp"), emissions.EUROPE)
    assert emissions.Header.unpack(header.pack()) == header
    assert header.frame(11, 6, 15) == 12 * 7 * 16 - 1
    with pytest.raises(ValueError, match="version 1"):
        emissions.Header.unpack(b"NOTEMISS" + header.pack()[8:])
    path = emissions.build(
        "black_carbon", 2021, tmp_path / "dist", tmp_path / "work", fetcher(tmp_path), box=BOX
    )
    data = bytearray(path.read_bytes())
    small = emissions.Header.unpack(bytes(data))
    start = emissions.OFFSET.unpack_from(data, small.size)[0]
    emissions.FRAME_HEADER.pack_into(data, start, 0, 1, 1, 0)
    with pytest.raises(ValueError, match="is not month 0"):
        emissions.read_tile(bytes(data), 0, 0, 0)


def test_a_file_without_twelve_months_of_sectors_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "short.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 3)
        dataset.createDimension("lat", LATS.size)
        dataset.createDimension("lon", LONS.size)
        dataset.createVariable("lat", "f8", ("lat",))[:] = LATS
        dataset.createVariable("lon", "f8", ("lon",))[:] = LONS
        dataset.createVariable("sum", "f4", ("time", "lat", "lon"))[:] = 0.0
        dataset.createVariable("res", "f4", ("time", "lat", "lon"))[:] = 0.0
    with pytest.raises(ValueError, match="12 months"):
        emissions.read(path, BOX)


def test_a_download_of_several_files_is_refused(tmp_path: Path) -> None:
    def fetch(_wanted: dict[str, list[str]], target: Path) -> None:
        with zipfile.ZipFile(target, "w") as archive:
            for name in ("a.nc", "b.nc"):
                archive.writestr(name, b"x")

    with pytest.raises(ValueError, match="one netCDF file"):
        emissions.build("black_carbon", 2021, tmp_path / "dist", tmp_path / "work", fetch)


def test_the_plan_lists_what_is_not_built() -> None:
    built = {f"emissions-{species}-2021.emissions" for species in emissions.SPECIES[:2]}
    assert emissions.plan([2020, 2021], built) == [
        {"species": "black_carbon", "year": 2020},
        {"species": "organic_carbon", "year": 2020},
        {"species": "nitrogen_oxides", "year": 2020},
        {"species": "nitrogen_oxides", "year": 2021},
    ]


def test_the_build_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_build(species: str, year: int, out: Path, workdir: Path) -> Path:
        return out / f"{species}-{year}-{workdir.name}"

    monkeypatch.setattr(emissions, "build", fake_build)
    built = tmp_path / "built.txt"
    built.write_text("emissions-black_carbon-2021.emissions", encoding="utf-8")
    assert cli.main(["emissions-plan", "--years", "2021", "--built", str(built)]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2
    assert cli.main(["emissions-plan"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 3 * 14
    arguments = [
        "--species",
        "nitrogen_oxides",
        "--year",
        "2020",
        "--out",
        "dist",
        "--workdir",
        "w",
    ]
    assert cli.main(["emissions-build", *arguments]) == 0
    assert capsys.readouterr().out.strip().endswith("nitrogen_oxides-2020-w")
