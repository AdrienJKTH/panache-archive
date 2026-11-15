import json
import urllib.request
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from panache_archive import live_stations, stations
from panache_archive.build import read_metadata
from panache_archive.live_stations import Screening
from panache_archive.stations import Pollutant
from test_stations import POLLUTANT_URL, hours, listing, metadata, station_file


def times(count: int, start: str = "2026-11-10T00") -> np.ndarray:
    return np.datetime64(start, "s") + np.arange(count) * np.timedelta64(1, "h")


def test_negative_values_are_removed_and_counted_on_their_day() -> None:
    screening = Screening(Pollutant.PM2_5)

    kept = screening(times(3), np.array([5.0, -1.0, 0.0]))

    assert kept.tolist() == [True, False, True]
    assert screening.of_day("2026-11-10") == {"negative": 1, "ceiling": 0, "stuck": 0, "kept": 2}


def test_values_above_the_pollutant_s_ceiling_are_removed() -> None:
    screening = Screening(Pollutant.OZONE)

    kept = screening(times(2), np.array([800.0, 801.0]))

    assert kept.tolist() == [True, False]
    assert screening.total()["ceiling"] == 1


def test_a_value_unchanged_for_a_whole_day_is_stuck() -> None:
    values = np.array([7.0] * 24 + [8.0] + [9.0] * 23)
    screening = Screening(Pollutant.PM10)

    kept = screening(times(values.size), values)

    assert kept.tolist() == [False] * 24 + [True] * 24
    assert screening.total() == {"negative": 0, "ceiling": 0, "stuck": 24, "kept": 24}


def test_a_gap_in_the_hours_breaks_a_run() -> None:
    hours_ = np.concatenate([times(12), times(12, "2026-11-10T13")])

    assert not live_stations.stuck(hours_, np.full(24, 3.0)).any()
    assert live_stations.stuck(hours_[:0], np.array([])).size == 0


def test_an_hour_counts_for_the_first_rule_that_removes_it() -> None:
    screening = Screening(Pollutant.PM2_5)

    screening(times(24), np.full(24, -2.0))

    assert screening.total() == {"negative": 24, "ceiling": 0, "stuck": 0, "kept": 0}


def test_the_screen_runs_before_the_daily_statistic() -> None:
    # 20 good hours and 4 negative ones: 20 hours remain, enough for the day.
    rows = hours(datetime(2026, 11, 10, 1), [9.0, 11.0] * 10 + [-5.0] * 4)

    found = stations.daily_means(station_file("SPO-X", rows), screen=Screening(Pollutant.PM2_5))

    assert found.means.tolist() == [10.0]
    assert found.hours.tolist() == [20]


def test_a_reported_day_s_value_is_screened_too() -> None:
    rows: list[tuple[datetime, float | None, str, str, int]] = [
        (datetime(2026, 11, 10), -3.0, "ug.m-3", "day", 1)
    ]

    found = stations.daily_means(station_file("SPO-X", rows), screen=Screening(Pollutant.PM10))

    assert found.days.size == 0


def point_row(country: str, point: str, station: str, code: str) -> str:
    return (
        f"{country}\t{station}\t{point}\t{POLLUTANT_URL}{code}\t6.1\t49.6\t300\tbackground\turban"
    )


def fake_service(files: dict[str, bytes], requests: list[dict[str, object]]) -> stations.Read:
    meta = metadata(
        point_row("LU", "SPO-LU1", "LU0001A", "6001"),
        point_row("MT", "SPO-MT1", "MT0001A", "6001"),
    )

    def read(request: urllib.request.Request | str) -> bytes:
        url = request if isinstance(request, str) else request.full_url
        if url.endswith("/Country"):
            return json.dumps([{"countryCode": "MT"}, {"countryCode": "LU"}]).encode()
        if url == stations.METADATA_URL:
            return meta
        if url.endswith("/ParquetFile/urls"):
            assert isinstance(request, urllib.request.Request)
            body = json.loads(request.data or b"{}")  # type: ignore[arg-type]
            requests.append(body)
            country = body["countries"][0]
            if body["aggregationType"] != "hour":
                return listing()
            return listing(*(u for u in files if f"/{country}/" in u))
        return files[url]

    return read


def test_a_day_s_asset_holds_every_country_s_points_in_the_validated_schema(
    tmp_path: Path,
) -> None:
    day = [9.0, 11.0] * 12
    files = {
        "https://e/LU/a.parquet": station_file("SPO-LU1", hours(datetime(2026, 11, 10, 1), day)),
        "https://e/MT/b.parquet": station_file("SPO-MT1", hours(datetime(2026, 11, 10, 1), day)),
        "https://e/MT/c.parquet": station_file("SPO-MT9", hours(datetime(2026, 11, 10, 1), day)),
    }
    requests: list[dict[str, object]] = []

    paths = live_stations.build(
        Pollutant.PM2_5,
        date(2026, 11, 10),
        date(2026, 11, 11),
        tmp_path,
        read=fake_service(files, requests),
        downloads=1,
    )

    assert [p.name for p in paths] == [
        "stations-live-pm2_5-2026-11-10.parquet",
        "stations-live-pm2_5-2026-11-11.parquet",
    ]
    assert {r["dataset"] for r in requests} == {live_stations.UP_TO_DATE}
    table = pq.read_table(paths[0])
    assert table.column_names == ["sampling_point", "day", "mean", "hours"]
    assert table["mean"].to_pylist() == [10.0, 10.0]
    described = read_metadata(paths[0])
    assert described["source"] == "eea-e2a"
    assert described["countries"] == ["LU", "MT"]
    assert described["left_out"] == 1  # SPO-MT9 is not in the metadata
    assert [p["country"] for p in described["points"]] == ["LU", "MT"]
    assert described["screening"]["removed"]["kept"] == 72
    assert pq.read_table(paths[1]).num_rows == 0
    report = json.loads((tmp_path / "screening-pm2_5.json").read_text())
    assert report == {"negative": 0, "ceiling": 0, "stuck": 0, "kept": 72, "files": 3}


def test_countries_can_be_chosen(tmp_path: Path) -> None:
    requests: list[dict[str, object]] = []

    live_stations.build(
        Pollutant.OZONE,
        date(2026, 11, 10),
        date(2026, 11, 10),
        tmp_path,
        countries=["LU"],
        read=fake_service({}, requests),
    )

    assert {tuple(r["countries"]) for r in requests} == {("LU",)}  # type: ignore[arg-type]
    assert live_stations.asset_name(Pollutant.OZONE, date(2026, 1, 2)) == (
        "stations-live-ozone-2026-01-02.parquet"
    )
