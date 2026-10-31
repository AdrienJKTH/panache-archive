import asyncio
import json
import re
import urllib.parse
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from numpy.typing import NDArray

from panache_archive import cells, era5
from panache_archive.build import read_metadata
from panache_archive.check import Place, compare_twin
from panache_archive.daily import daily_means
from panache_archive.domain import Domain
from panache_archive.era5 import Daily, Era5Job, Hourly
from panache_archive.grid import Box

FloatArray = NDArray[np.float64]
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
LYON = Place("Lyon", 45.76, 4.84)
SMALL = Box(south=45.5, north=46.0, west=4.5, east=5.25)
"""Around Lyon: 3 by 4 ERA5 points."""
ALTITUDE_OFFSET = 1.22
"""What the archive API adds to temperature and dew point at the fake Lyon."""


def hour_of(day: date) -> int:
    return (datetime(day.year, day.month, day.day, tzinfo=UTC) - EPOCH) // timedelta(hours=1)


def synthetic(variable: Hourly, hours: NDArray[np.int64], y: int, x: int) -> FloatArray:
    """Hourly values of `variable` at grid point (y, x), by hour since 1970."""
    cycle = np.sin(2 * np.pi * (hours % 24) / 24)
    slow = np.sin(2 * np.pi * hours / (24 * 7.3))
    place = (y - 540) / 10 + (x - 740) / 20
    values = {
        Hourly.TEMPERATURE: 10 + 6 * cycle + 4 * slow + place,
        Hourly.DEW_POINT: 5 + 2 * cycle + 3 * slow + place,
        Hourly.WIND_U: 3 * slow + 0.5 * cycle,
        Hourly.WIND_V: -2 + 2 * cycle,
        Hourly.PRESSURE: 101_300 + 800 * slow,
        Hourly.PRECIPITATION: np.maximum(0, 2 * slow - 0.5) * (hours % 3 == 0),
        Hourly.RADIATION: np.maximum(0, 400 * cycle),
        Hourly.CLOUD_COVER: 50 + 40 * slow,
        Hourly.BOUNDARY_LAYER_HEIGHT: 600 + 400 * cycle + 100 * slow,
    }[variable]
    return np.asarray(values, dtype=np.float64)


class FakeBucket:
    """Serves `synthetic` values, from year files for `year_files` and from chunks otherwise."""

    def __init__(
        self, year_files: set[int] | None = None, short: bool = False, gaps: set[str] | None = None
    ) -> None:
        self.year_files = year_files or set()
        self.short = short
        self.gaps = gaps or set()
        """Chunk files the bucket lacks."""
        self.reads: list[str] = []

    async def exists(self, path: str) -> bool:
        found = re.search(r"year_(\d{4})\.om$", path)
        if found is None:
            return path not in self.gaps
        return int(found.group(1)) in self.year_files

    async def read(self, path: str, rows: slice, cols: slice, hours: slice) -> FloatArray:
        self.reads.append(path)
        variable, name = path.split("/")
        year = re.fullmatch(r"year_(\d{4})\.om", name)
        chunk = re.fullmatch(r"chunk_(\d+)\.om", name)
        if year is not None:
            base = hour_of(date(int(year.group(1)), 1, 1))
        else:
            assert chunk is not None
            base = int(chunk.group(1)) * era5.CHUNK_HOURS
        stop = hours.stop - (1 if self.short else 0)
        absolute = base + np.arange(hours.start, stop)
        return np.stack(
            [
                np.stack(
                    [
                        synthetic(Hourly(variable), absolute, y, x)
                        for x in range(cols.start, cols.stop)
                    ]
                )
                for y in range(rows.start, rows.stop)
            ]
        )


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


LYON_ROWS, LYON_COLS = slice(543, 544), slice(739, 740)


def test_year_files_and_chunks_give_the_same_hours() -> None:
    job = Era5Job(2021, (1,))
    from_year = run(era5.hourly(FakeBucket({2021}), Hourly.TEMPERATURE, job, LYON_ROWS, LYON_COLS))
    chunks = FakeBucket()
    from_chunks = run(era5.hourly(chunks, Hourly.TEMPERATURE, job, LYON_ROWS, LYON_COLS))

    assert from_year.shape == (31 * 24, 1, 1)
    np.testing.assert_array_equal(from_year, from_chunks)
    first = hour_of(date(2021, 1, 1))
    expected = synthetic(Hourly.TEMPERATURE, first + np.arange(31 * 24), 543, 739)
    np.testing.assert_array_equal(from_year[:, 0, 0], expected)
    # 2021-01-01 is hour 447,072 since 1970: chunk 887 (from hour 447,048) and chunk 888.
    assert chunks.reads == [f"temperature_2m/chunk_{n}.om" for n in (887, 888)]


def test_accumulations_take_the_hours_ending_in_each_day() -> None:
    bucket = FakeBucket({2021, 2022})
    job = Era5Job(2021, (12,))

    rain = run(era5.hourly(bucket, Hourly.PRECIPITATION, job, LYON_ROWS, LYON_COLS))

    first = hour_of(date(2021, 12, 1)) + 1
    expected = synthetic(Hourly.PRECIPITATION, first + np.arange(31 * 24), 543, 739)
    np.testing.assert_array_equal(rain[:, 0, 0], expected)
    # The last hour, 2022-01-01 00:00, comes from the next year's file.
    assert bucket.reads == ["precipitation/year_2021.om", "precipitation/year_2022.om"]


def test_missing_hours_are_an_error() -> None:
    with pytest.raises(ValueError, match="hours, expected 744"):
        run(
            era5.hourly(
                FakeBucket({2021}, short=True),
                Hourly.CLOUD_COVER,
                Era5Job(2021, (1,)),
                LYON_ROWS,
                LYON_COLS,
            )
        )


def test_daily_weather_follows_the_archive_api() -> None:
    hours = 48
    values = {
        Hourly.TEMPERATURE: np.full((hours, 1, 1), 20.0),
        Hourly.DEW_POINT: np.full((hours, 1, 1), 20.0),
        Hourly.WIND_U: np.full((hours, 1, 1), -3.0),
        Hourly.WIND_V: np.full((hours, 1, 1), -4.0),
        Hourly.PRESSURE: np.full((hours, 1, 1), 101_325.0),
        Hourly.PRECIPITATION: np.full((hours, 1, 1), 0.5),
        Hourly.RADIATION: np.full((hours, 1, 1), 100.0),
        Hourly.CLOUD_COVER: np.full((hours, 1, 1), 75.0),
        Hourly.BOUNDARY_LAYER_HEIGHT: np.full((hours, 1, 1), 450.0),
    }

    daily = era5.daily_weather(values)

    assert {c: float(v[0, 0, 0]) for c, v in daily.items()} == pytest.approx(
        {
            Daily.TEMPERATURE: 20.0,
            Daily.RELATIVE_HUMIDITY: 100.0,
            Daily.WIND_SPEED: 5.0,
            Daily.WIND_DIRECTION: np.degrees(np.arctan2(3, 4)),  # from the north-east
            Daily.PRESSURE: 1013.25,
            Daily.PRECIPITATION: 12.0,
            Daily.RADIATION: 8.64,
            Daily.CLOUD_COVER: 75.0,
            Daily.BOUNDARY_LAYER_HEIGHT: 450.0,
        }
    )


@pytest.mark.parametrize(
    ("u", "v", "direction"),
    [(0.0, -5.0, 0.0), (-5.0, 0.0, 90.0), (0.0, 5.0, 180.0), (5.0, 0.0, 270.0)],
)
def test_the_wind_direction_is_where_it_blows_from(u: float, v: float, direction: float) -> None:
    assert era5.wind_direction(np.array([u]), np.array([v]))[0] == pytest.approx(direction)


def test_humidity_is_lower_when_the_dew_point_is_lower() -> None:
    humidity = era5.relative_humidity(np.array([20.0, 20.0]), np.array([20.0, 10.0]))

    assert humidity[0] == pytest.approx(100.0)
    assert humidity[1] == pytest.approx(52.5, abs=0.5)


def test_an_asset_holds_the_box_each_day(tmp_path: Path) -> None:
    path = run(era5.build(Era5Job(2021, (1,)), tmp_path, FakeBucket({2021}), box=SMALL))

    assert path.name == "era5-2021-01.parquet"
    table = pq.read_table(path)
    assert table.column_names == ["lat", "lon", "day", *(c.value for c in Daily)]
    assert table.num_rows == 3 * 4 * 31
    assert np.unique(table["lat"].to_numpy()).tolist() == [45.5, 45.75, 46.0]
    assert np.unique(table["lon"].to_numpy()).tolist() == [4.5, 4.75, 5.0, 5.25]
    lyon = table.filter(
        (table["lat"].to_numpy() == 45.75) & (table["lon"].to_numpy() == 4.75)
    ).sort_by("day")
    first = hour_of(date(2021, 1, 1))
    temperature = synthetic(Hourly.TEMPERATURE, first + np.arange(31 * 24), 543, 739)
    np.testing.assert_allclose(
        lyon[Daily.TEMPERATURE.value].to_numpy(), np.round(daily_means(temperature), 2), atol=1e-4
    )
    metadata = json.loads(pq.read_schema(path).metadata[b"panache_archive"])
    assert metadata["source"] == "era5"
    assert metadata["units"]["shortwave_radiation_sum"] == "MJ/m²"
    assert "Copernicus Climate Change Service" in metadata["attribution"]


class FakeArchiveApi:
    """The archive API at Lyon's point (543, 739), from `synthetic`, as the API makes its days:
    temperature and dew point corrected for altitude, values rounded."""

    def __init__(
        self,
        units: dict[str, str] | None = None,
        days_short: int = 0,
        rain_hour_earlier: bool = False,
    ) -> None:
        self.units = units or {}
        self.days_short = days_short
        self.rain_hour_earlier = rain_hour_earlier
        """Sum the rain over 00:00 to 23:00, as the API does since October 2026 (D62)."""
        self.queries: list[dict[str, str]] = []

    def __call__(self, url: str) -> bytes:
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        self.queries.append(query)
        first_day = date.fromisoformat(query["start_date"])
        days = (date.fromisoformat(query["end_date"]) - first_day).days + 1
        first = hour_of(first_day)
        values = {
            v: synthetic(v, first + np.arange(days * 24) + self._shift(v), 543, 739)[:, None, None]
            for v in Hourly
        }
        values[Hourly.TEMPERATURE] = values[Hourly.TEMPERATURE] + ALTITUDE_OFFSET
        values[Hourly.DEW_POINT] = values[Hourly.DEW_POINT] + ALTITUDE_OFFSET
        daily = era5.daily_weather(values)
        decimals = {Daily.WIND_DIRECTION: 0, Daily.CLOUD_COVER: 0, Daily.RELATIVE_HUMIDITY: 0}
        shown = days - self.days_short
        answer: dict[str, Any] = {
            "daily": {
                c.value: [round(float(v), decimals.get(c, 1)) for v in daily[c][:shown, 0, 0]]
                for c in Daily
                if c is not Daily.BOUNDARY_LAYER_HEIGHT
            },
            "daily_units": {c.value: self.units.get(c.value, era5.UNITS[c]) for c in Daily},
            "hourly": {
                Daily.BOUNDARY_LAYER_HEIGHT.value: [
                    round(float(v)) for v in values[Hourly.BOUNDARY_LAYER_HEIGHT][:, 0, 0]
                ]
            },
            "hourly_units": {Daily.BOUNDARY_LAYER_HEIGHT.value: "m"},
        }
        return json.dumps(answer).encode()

    def _shift(self, variable: Hourly) -> int:
        if variable is Hourly.PRECIPITATION and self.rain_hour_earlier:
            return 0
        return 1 if variable in era5.ACCUMULATED else 0


@pytest.fixture
def asset(tmp_path: Path) -> Path:
    path: Path = run(era5.build(Era5Job(2021, (1,)), tmp_path, FakeBucket({2021, 2022}), box=SMALL))
    return path


def test_an_asset_agrees_with_the_archive_api(asset: Path) -> None:
    api = FakeArchiveApi()

    comparisons = era5.compare(asset, [LYON], get=api)

    assert [c.column for c in comparisons] == [c.value for c in Daily]
    assert all(c.cell == (45.75, 4.75) for c in comparisons)
    assert [c.column for c in comparisons if not c.agrees] == []
    temperature = comparisons[0]
    assert temperature.offset == pytest.approx(-ALTITUDE_OFFSET, abs=0.06)
    assert api.queries[0]["models"] == "era5"
    assert api.queries[0]["hourly"] == "boundary_layer_height"
    assert api.queries[0]["start_date"] == "2021-01-01"
    assert api.queries[0]["end_date"] == "2021-01-31"


def test_another_unit_is_an_error(asset: Path) -> None:
    with pytest.raises(ValueError, match="km/h"):
        era5.compare(asset, [LYON], get=FakeArchiveApi(units={"wind_speed_10m_mean": "km/h"}))


def test_a_short_answer_is_an_error(asset: Path) -> None:
    with pytest.raises(ValueError, match="answered 30 values"):
        era5.compare(asset, [LYON], get=FakeArchiveApi(days_short=1))


def test_a_cams_asset_is_not_compared_as_era5(tmp_path: Path) -> None:
    from panache_archive.build import to_table

    path = tmp_path / "cams.parquet"
    pq.write_table(
        to_table(
            np.array([45.75]),
            np.array([4.85]),
            date(2021, 2, 1),
            {"mean": np.ones((1, 1, 1))},
            metadata={"variable": "pm10"},
        ),
        path,
    )

    with pytest.raises(ValueError, match="not an ERA5 asset"):
        era5.compare(path, [LYON])


@pytest.mark.parametrize(
    ("today", "year"),
    [(date(2026, 10, 17), 2025), (date(2026, 1, 9), 2024), (date(2026, 1, 10), 2025)],
)
def test_the_last_complete_year_waits_for_era5_to_catch_up(today: date, year: int) -> None:
    assert era5.last_complete_year(today) == year


def test_jobs_and_names() -> None:
    assert Era5Job(2021).asset_name == "era5-2021.parquet"
    assert Era5Job(2024, (2,)).days == 29
    assert Era5Job(2021).days == 365
    assert [j.year for j in era5.missing_years([2020, 2021], {"era5-2020.parquet"})] == [2021]
    with pytest.raises(ValueError, match="months"):
        Era5Job(2021, (3, 1))


def test_the_http_file_system_takes_omfiles_argument_names(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any]] = []

    async def cat_file(
        _self: object, url: str, start: int | None = None, end: int | None = None
    ) -> bytes:
        calls.append(("cat", (url, start, end)))
        return b"bytes"

    async def size(_self: object, url: str) -> int:
        calls.append(("size", url))
        return 7

    monkeypatch.setattr("fsspec.implementations.http.HTTPFileSystem._cat_file", cat_file)
    monkeypatch.setattr("fsspec.implementations.http.HTTPFileSystem._size", size)
    fs = era5._Http(asynchronous=True)

    assert run(fs._cat_file(path="u", start=1, end=5)) == b"bytes"
    assert run(fs._size(path="u")) == 7
    assert calls == [("cat", ("u", 1, 5)), ("size", "u")]


def test_the_http_bucket_reads_through_omfiles(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []

    class Reader:
        closed = False

        async def read_array(self, ranges: tuple[slice, slice, slice]) -> NDArray[np.float32]:
            rows, cols, hours = ranges
            return np.ones(
                (rows.stop - rows.start, cols.stop - cols.start, hours.stop - hours.start),
                np.float32,
            )

        def close(self) -> None:
            Reader.closed = True

    class OmFileReaderAsync:
        @staticmethod
        async def from_fsspec(_fs: object, path: str) -> Reader:
            opened.append(path)
            return Reader()

    class Session:
        closed = False

        async def close(self) -> None:
            Session.closed = True

    async def set_session(_self: object) -> Session:
        return Session()

    async def exists(_self: object, path: str) -> bool:
        return path.endswith("year_2021.om")

    monkeypatch.setattr("panache_archive.era5.OmFileReaderAsync", OmFileReaderAsync)
    monkeypatch.setattr("panache_archive.era5._Http.set_session", set_session)
    monkeypatch.setattr("panache_archive.era5._Http._exists", exists, raising=False)

    async def use() -> tuple[bool, bool, FloatArray]:
        async with era5.HttpBucket("https://bucket") as bucket:
            found = await bucket.exists("temperature_2m/year_2021.om")
            missing = await bucket.exists("temperature_2m/year_2024.om")
            values = await bucket.read(
                "temperature_2m/year_2021.om", slice(0, 2), slice(0, 3), slice(0, 4)
            )
            return found, missing, values

    found, missing, values = run(use())

    assert (found, missing) == (True, False)
    assert values.shape == (2, 3, 4)
    assert values.dtype == np.float64
    assert opened == ["https://bucket/temperature_2m/year_2021.om"]
    assert Reader.closed
    assert Session.closed


def test_a_chunk_the_bucket_lacks_leaves_its_hours_missing() -> None:
    # Like the boundary layer of early 2024, which Open-Meteo does not have either.
    bucket = FakeBucket(gaps={"boundary_layer_height/chunk_887.om"})
    job = Era5Job(2021, (1,))

    values = run(era5.hourly(bucket, Hourly.BOUNDARY_LAYER_HEIGHT, job, LYON_ROWS, LYON_COLS))

    # Chunk 887 ends at hour 447,552: the first 480 hours of January 2021.
    assert np.isnan(values[:480]).all()
    assert not np.isnan(values[480:]).any()
    assert bucket.reads == ["boundary_layer_height/chunk_888.om"]


def test_a_variable_wholly_missing_is_an_error() -> None:
    gaps = {"cloud_cover/chunk_887.om", "cloud_cover/chunk_888.om"}

    with pytest.raises(ValueError, match="none of the 744 hours"):
        run(
            era5.hourly(
                FakeBucket(gaps=gaps), Hourly.CLOUD_COVER, Era5Job(2021, (1,)), LYON_ROWS, LYON_COLS
            )
        )


def test_a_whole_domain_asset_holds_france_s_numbers_one_point_a_frame(tmp_path: Path) -> None:
    job = Era5Job(2021, (1,), Domain.EUROPE)
    france = run(era5.build(Era5Job(2021, (1,)), tmp_path, FakeBucket({2021}), box=SMALL))

    path = run(era5.build(job, tmp_path, FakeBucket({2021}), box=SMALL, band_rows=2))

    assert path.name == "era5-2021-01.cells"
    header, hundredths = cells.read_all(path)
    assert (header.rows, header.columns, header.days, header.statistics) == (3, 4, 31, 9)
    assert (header.cells_per_frame, header.step, header.first_lat) == (1, 0.25, 45.5)
    twin = compare_twin(path, france)
    assert (twin.cells, twin.days, twin.differing) == (12, 31, 0)
    metadata = read_metadata(path)
    assert (metadata["source"], metadata["domain"]) == ("era5", "europe")
    assert metadata["columns"] == [c.value for c in Daily]
    assert hundredths.shape == (9, 31, 3, 4)


def test_a_whole_domain_asset_agrees_with_the_archive_api(tmp_path: Path) -> None:
    job = Era5Job(2021, (1,), Domain.EUROPE)
    path = run(era5.build(job, tmp_path, FakeBucket({2021, 2022}), box=SMALL))

    comparisons = era5.compare(path, [LYON], get=FakeArchiveApi())

    assert [c.column for c in comparisons if not c.agrees] == []
    assert all(c.cell == (45.75, 4.75) for c in comparisons)


def test_bands_change_nothing_and_must_hold_a_row(tmp_path: Path) -> None:
    one = run(era5.build(Era5Job(2021, (1,)), tmp_path / "one", FakeBucket({2021}), box=SMALL))
    banded = run(
        era5.build(Era5Job(2021, (1,)), tmp_path / "b", FakeBucket({2021}), box=SMALL, band_rows=1)
    )

    assert pq.read_table(one).drop(["day"]).equals(pq.read_table(banded).drop(["day"]))
    with pytest.raises(ValueError, match="hold nothing"):
        run(era5.build(Era5Job(2021, (1,)), tmp_path, FakeBucket({2021}), box=SMALL, band_rows=0))


def test_whole_domain_jobs_are_named_cells() -> None:
    assert Era5Job(2021, domain=Domain.EUROPE).asset_name == "era5-2021.cells"
    assert [
        j.asset_name for j in era5.missing_years([2020, 2021], {"era5-2020.cells"}, Domain.EUROPE)
    ] == ["era5-2021.cells"]


def test_rain_summed_an_hour_earlier_still_agrees_by_correlation(asset: Path) -> None:
    comparisons = era5.compare(asset, [LYON], get=FakeArchiveApi(rain_hour_earlier=True))

    (rain,) = [c for c in comparisons if c.column == Daily.PRECIPITATION.value]
    assert rain.max_difference > rain.tolerance  # the days differ by an hour of rain
    assert rain.min_correlation == era5.PRECIPITATION_CORRELATION
    assert rain.agrees
