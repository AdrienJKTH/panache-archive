import json
import urllib.error
import urllib.parse
from datetime import date, timedelta
from email.message import Message
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from panache_archive import cells, merra2
from panache_archive.build import read_metadata
from panache_archive.check import Place
from panache_archive.merra2 import LATS, LONS, SPECIES, Merra2Job

LYON = Place("Lyon", 45.76, 4.84)
"""MERRA-2's cell (46.0, 5.0): row 32, column 48."""


def hourly(species: str, day: date) -> np.ndarray:
    """Synthetic hours of `species` on `day`, `(hour, lat, lon)` in kg/m³: a value in μg/m³ of
    `k + row / 100 + column / 1000 + day of year / 10 + hour / 100`."""
    k = SPECIES.index(species)
    rows = np.arange(LATS.size)[None, :, None]
    cols = np.arange(LONS.size)[None, None, :]
    hours = np.arange(24)[:, None, None]
    ug = 1 + k + rows / 100 + cols / 1000 + day.timetuple().tm_yday / 10 + hours / 100
    return np.asarray(ug / merra2.UG_PER_KG, dtype=np.float32)


def expected(species: str, day: date, row: int, column: int) -> float:
    return float(hourly(species, day)[:, row, column].astype(np.float64).mean() * merra2.UG_PER_KG)


def netcdf_day(
    day: date, *, lats: np.ndarray = LATS, units: str = "kg m-3", gap_hours: int = 0
) -> bytes:
    """The answer OPeNDAP gives for the domain's cut of `day`."""
    dataset = netCDF4.Dataset("day.nc4", "w", memory=0, format="NETCDF4")
    dataset.createDimension("time", 24)
    dataset.createDimension("lat", lats.size)
    dataset.createDimension("lon", LONS.size)
    dataset.createVariable("lat", "f8", ("lat",))[:] = lats
    dataset.createVariable("lon", "f8", ("lon",))[:] = LONS
    for species in SPECIES:
        variable = dataset.createVariable(species, "f4", ("time", "lat", "lon"), fill_value=1e15)
        variable.units = units
        values = np.ma.masked_array(hourly(species, day)[:, : lats.size])
        if gap_hours:
            values[:gap_hours, 0, 0] = np.ma.masked
        variable[:] = values
    return bytes(dataset.close())


class FakeOpenDap:
    """Serves `netcdf_day` for each day's URL of stream 400, or 401 for `reprocessed` days; 404
    for days after `last_day`."""

    def __init__(self, last_day: date = date(2026, 9, 30), reprocessed: set[date] | None = None):
        self.last_day = last_day
        self.reprocessed = reprocessed or set()
        self.asked: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.asked.append(url)
        granule = urllib.parse.unquote(url.split("/granules/")[1].split("?")[0])
        stream = granule.split("MERRA2_")[1][:3]
        day = date.fromisoformat(granule.split("Nx.")[1][:8])
        wanted = "401" if day in self.reprocessed else "400"
        if day > self.last_day or stream != wanted:
            raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)
        if granule.endswith(".dds"):
            return b"Dataset {} day;"
        return netcdf_day(day)


def test_a_day_is_cut_to_the_domain_and_averaged() -> None:
    day = date(2021, 2, 6)

    means = merra2.read_day(day, FakeOpenDap())

    assert means.shape == (3, 85, 113)
    for k, species in enumerate(SPECIES):
        assert means[k, 32, 48] == pytest.approx(expected(species, day, 32, 48), rel=1e-6)


def test_a_reprocessed_month_is_read_from_its_401_file() -> None:
    day = date(2020, 9, 15)
    opendap = FakeOpenDap(reprocessed={day})

    merra2.read_day(day, opendap)

    assert ["MERRA2_400" in u for u in opendap.asked] == [True, False]


def test_a_day_with_too_few_hours_is_missing() -> None:
    day = date(2021, 2, 6)

    def gappy(_url: str) -> bytes:
        return netcdf_day(day, gap_hours=7)

    means = merra2.read_day(day, gappy)

    assert np.isnan(means[:, 0, 0]).all()
    assert not np.isnan(means[:, 0, 1]).any()


@pytest.mark.parametrize(
    ("answer", "error"),
    [
        (lambda day: netcdf_day(day, lats=LATS[:-1]), "not the domain's grid"),
        (lambda day: netcdf_day(day, units="ug m-3"), "in ug m-3"),
    ],
)
def test_an_odd_cut_is_an_error(answer: object, error: str) -> None:
    day = date(2021, 2, 6)

    with pytest.raises(ValueError, match=error):
        merra2.read_day(day, lambda _url: answer(day))  # type: ignore[operator]


def test_a_missing_day_is_an_error() -> None:
    with pytest.raises(urllib.error.HTTPError):
        merra2.read_day(date(2026, 10, 15), FakeOpenDap())


def test_an_asset_holds_each_species_of_each_day(tmp_path: Path) -> None:
    job = Merra2Job(2021, (2,))

    path = merra2.build(job, tmp_path, FakeOpenDap(), concurrency=4)

    assert path.name == "merra2-2021-02.cells"
    header = cells.read_header(path)
    assert (header.rows, header.columns, header.days, header.statistics) == (85, 113, 28, 3)
    assert (header.step, header.longitude_step, header.cells_per_frame) == (0.5, 0.625, 1)
    assert (header.first_lat, header.first_lon) == (30.0, -25.0)
    np.testing.assert_array_equal(header.lons, LONS)
    values = cells.to_values(cells.read_cell(path, 32, 48), header.scale)
    for k, species in enumerate(SPECIES):
        assert values[k, 5] == pytest.approx(expected(species, date(2021, 2, 6), 32, 48), abs=0.006)
    metadata = read_metadata(path)
    assert (metadata["source"], metadata["statistics"]) == ("merra2", list(SPECIES))
    assert metadata["doi"] == "10.5067/KLICLTZ8EM9D"


def test_the_last_month_out_is_probed_from_the_latest() -> None:
    opendap = FakeOpenDap(last_day=date(2026, 9, 30))

    assert merra2.last_month_out(2026, date(2026, 10, 31), opendap) == 9
    assert merra2.last_month_out(2025, date(2026, 10, 31), opendap) == 12
    assert merra2.last_month_out(2026, date(2026, 1, 20), FakeOpenDap(date(2025, 12, 31))) == 0
    assert all(u.endswith(".dds") for u in opendap.asked)


def test_the_plan_builds_whole_years_and_the_year_under_way() -> None:
    built = {"merra2-2013.cells", "merra2-2026-01-08.cells", "merra2-2025.cells"}

    jobs, superseded = merra2.plan(range(2013, 2027), built, date(2026, 10, 31), FakeOpenDap())

    assert [j.asset_name for j in jobs] == [
        *(f"merra2-{y}.cells" for y in range(2014, 2025)),
        "merra2-2026-01-09.cells",
    ]
    assert superseded == ["merra2-2026-01-08.cells"]


def test_a_year_once_whole_supersedes_its_part() -> None:
    built = {"merra2-2026-01-11.cells"}

    jobs, superseded = merra2.plan(
        [2026], built, date(2027, 2, 10), FakeOpenDap(date(2026, 12, 31))
    )

    assert [j.asset_name for j in jobs] == ["merra2-2026.cells"]
    assert superseded == ["merra2-2026-01-11.cells"]


def test_nothing_out_yet_plans_nothing() -> None:
    jobs, superseded = merra2.plan(
        [2027], set(), date(2027, 1, 20), FakeOpenDap(date(2026, 12, 31))
    )

    assert (jobs, superseded) == ([], [])


class FakeGiovanni:
    """Giovanni's CSV of `hourly` at the cell nearest the asked place."""

    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset

    def __call__(self, url: str) -> bytes:
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        species = query["data"].rsplit("_", 1)[1]
        lat, lon = (float(x) for x in query["location"].strip("[]").split(","))
        row, column = int(np.argmin(abs(LATS - lat))), int(np.argmin(abs(LONS - lon)))
        start, end = query["time"].split("/")
        first, last = date.fromisoformat(start[:10]), date.fromisoformat(end[:10])
        lines = [f"param_short_name,{species}", "unit,kg m-3", "undef,1e+15", ""]
        lines.append("Timestamp (UTC),Data")
        for n in range((last - first).days + 1):
            day = first + timedelta(days=n)
            for hour, value in enumerate(hourly(species, day)[:, row, column]):
                lines.append(f"{day}T{hour:02d}:30:00,{float(value) + self.offset / 1e9:.6e}")
        return "\n".join(lines).encode()


def test_an_asset_agrees_with_giovanni(tmp_path: Path) -> None:
    path = merra2.build(Merra2Job(2021, (2,)), tmp_path, FakeOpenDap())

    comparisons = merra2.compare(path, [LYON], fetch=FakeGiovanni())

    assert [c.column for c in comparisons] == list(SPECIES)
    assert all(c.cell == (46.0, 5.0) and c.days == 28 and c.agrees for c in comparisons)


def test_an_asset_off_giovanni_disagrees(tmp_path: Path) -> None:
    path = merra2.build(Merra2Job(2021, (2,)), tmp_path, FakeOpenDap())

    comparisons = merra2.compare(path, [LYON], fetch=FakeGiovanni(offset=0.1))

    assert not any(c.agrees for c in comparisons)


def test_comparing_needs_a_merra2_asset_and_the_token(tmp_path: Path) -> None:
    path = merra2.build(Merra2Job(2021, (2,)), tmp_path, FakeOpenDap())
    other = tmp_path / "other.cells"
    cells.write(
        other, LATS[:2], LONS[:2], date(2021, 1, 1), [np.ones((1, 2, 2))], metadata={"x": 1}
    )

    with pytest.raises(ValueError, match="token"):
        merra2.compare(path, [LYON])
    with pytest.raises(ValueError, match="not a MERRA-2 asset"):
        merra2.compare(other, [LYON], fetch=FakeGiovanni())


def test_names_and_days() -> None:
    assert Merra2Job(2021).asset_name == "merra2-2021.cells"
    assert Merra2Job(2026, tuple(range(1, 10))).asset_name == "merra2-2026-01-09.cells"
    assert Merra2Job(2024, (2,)).days == 29
    assert Merra2Job(2021).days == 365
    url = merra2.day_url("401", date(2020, 9, 15))
    assert "MERRA2_401.tavg1_2d_aer_Nx.20200915.nc4.nc4?DUSMASS" in urllib.parse.unquote(url)
    assert "[0:1:23][240:1:324][248:1:360]" in urllib.parse.unquote(url)
    assert json.dumps(merra2.SPECIES) == '["DUSMASS", "OCSMASS", "BCSMASS"]'
