import io
import json
import urllib.request
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from panache_archive import stations
from panache_archive.build import read_metadata
from panache_archive.stations import Pollutant

HEADER = (
    "Countrycode\tAirQualityStationEoICode\tSamplingPoint\tAirPollutantCode\tLongitude\t"
    "Latitude\tAltitude\tAirQualityStationType\tAirQualityStationArea\n"
)
POLLUTANT_URL = "http://dd.eionet.europa.eu/vocabulary/aq/pollutant/"


def station_file(point: str, rows: list[tuple[datetime, float | None, str, str, int]]) -> bytes:
    """A file as the service serves it: (start at UTC+1, value, unit, aggregation, validity)."""
    table = pa.table(
        {
            "Samplingpoint": pa.array([f"FR/{point}"] * len(rows)),
            "Pollutant": pa.array([6001] * len(rows), pa.int32()),
            "Start": pa.array([r[0] for r in rows], pa.timestamp("ns")),
            "End": pa.array([r[0] + timedelta(hours=1) for r in rows], pa.timestamp("ns")),
            "Value": pa.array(
                [None if r[1] is None else Decimal(str(r[1])) for r in rows],
                pa.decimal128(38, 18),
            ),
            "Unit": pa.array([r[2] for r in rows]),
            "AggType": pa.array([r[3] for r in rows]),
            "Validity": pa.array([r[4] for r in rows], pa.int32()),
        }
    )
    sink = io.BytesIO()
    pq.write_table(table, sink)
    return sink.getvalue()


def hours(
    first: datetime, values: list[float], validity: int = 1
) -> list[tuple[datetime, float | None, str, str, int]]:
    return [
        (first + timedelta(hours=i), value, "ug.m-3", "hour", validity)
        for i, value in enumerate(values)
    ]


def metadata(*rows: str) -> bytes:
    return (HEADER + "".join(row + "\n" for row in rows)).encode("utf-8-sig")


def listing(*urls: str) -> bytes:
    return ("﻿ParquetFileUrl\n" + "".join(url + "\n" for url in urls)).encode()


def test_a_day_is_the_mean_of_its_utc_hours() -> None:
    # The file's first hour starts at 01:00 UTC+1, midnight UTC: 24 hours make 6 February.
    rows = hours(datetime(2021, 2, 6, 1), [10.0] * 12 + [20.0] * 12)

    found = stations.daily_means(station_file("SPO-FR01011_6001", rows))

    assert found.sampling_point == "SPO-FR01011_6001"
    assert found.days.astype(str).tolist() == ["2021-02-06"]
    assert found.means.tolist() == [15.0]
    assert found.hours.tolist() == [24]


def test_an_hour_of_the_file_is_one_hour_earlier_in_utc() -> None:
    # 00:00 at UTC+1 is 23:00 UTC the day before: that day has one hour, too few.
    rows = hours(datetime(2021, 2, 6, 0), [5.0] * 19)

    found = stations.daily_means(station_file("SPO-FR01011_6001", rows))

    assert found.days.astype(str).tolist() == ["2021-02-06"]
    assert found.hours.tolist() == [18]


def test_days_need_18_valid_hours() -> None:
    valid = hours(datetime(2021, 2, 6, 1), [8.0] * 18)
    short = hours(datetime(2021, 2, 7, 1), [8.0] * 17)
    invalid = hours(datetime(2021, 2, 7, 19), [99.0] * 3, validity=-1)

    found = stations.daily_means(station_file("SPO-FR01011_6001", valid + short + invalid))

    assert found.days.astype(str).tolist() == ["2021-02-06"]


def test_other_units_aggregations_and_gaps_are_left_out() -> None:
    first = datetime(2021, 2, 6, 1)
    rows = [
        *hours(first, [10.0] * 18),
        (first + timedelta(hours=18), 500.0, "mg.m-3", "hour", 1),
        (first + timedelta(hours=19), 500.0, "ug.m-3", "day", 1),
        (first + timedelta(hours=20), None, "ug.m-3", "hour", 1),
    ]

    found = stations.daily_means(station_file("SPO-FR01011_6001", rows))

    assert found.means.tolist() == [10.0]
    assert found.hours.tolist() == [18]


def test_an_hour_held_twice_counts_once() -> None:
    rows = hours(datetime(2021, 2, 6, 1), [10.0] * 18)
    twice = hours(datetime(2021, 2, 6, 1), [70.0])

    found = stations.daily_means(station_file("SPO-FR01011_6001", rows + twice))

    assert found.means.tolist() == [10.0]
    assert found.hours.tolist() == [18]


def test_a_file_holds_one_point() -> None:
    first = station_file("SPO-FR01011_6001", hours(datetime(2021, 2, 6, 1), [1.0]))
    second = station_file("SPO-FR01012_6001", hours(datetime(2021, 2, 6, 2), [1.0]))
    tables = [pq.read_table(io.BytesIO(content)) for content in (first, second)]
    sink = io.BytesIO()
    pq.write_table(pa.concat_tables(tables), sink)

    with pytest.raises(ValueError, match="one sampling point"):
        stations.daily_means(sink.getvalue())


def test_points_come_from_the_metadata_once_each() -> None:
    content = metadata(
        f"FR\tFR01011\tSPO-FR01011_6001\t{POLLUTANT_URL}6001\t6.18\t49.12\t192\tbackground\turban",
        f"FR\tFR01011\tSPO-FR01011_6001\t{POLLUTANT_URL}6001\t9.99\t9.99\t1\ttraffic\turban",
        f"FR\tFR01012\tSPO-FR01012_6001\t{POLLUTANT_URL}6001\t2.35\t48.86\t-9999\ttraffic\turban",
        f"FR\tFR01013\tSPO-FR01013_6001\t{POLLUTANT_URL}6001\t\t\t30\tbackground\trural",
        f"FR\tFR01011\tSPO-FR01011_8\t{POLLUTANT_URL}8\t6.18\t49.12\t192\tbackground\turban",
        f"DE\tDEBB007\tSPO-DE_DEBB007\t{POLLUTANT_URL}6001\t13.0\t52.0\t40\tbackground\turban",
    )

    points = stations.read_points(content, "FR", Pollutant.PM2_5)

    assert list(points) == ["SPO-FR01011_6001", "SPO-FR01012_6001"]
    assert points["SPO-FR01011_6001"] == stations.Point(
        "SPO-FR01011_6001", "FR01011", 49.12, 6.18, 192.0, "background", "urban"
    )
    assert points["SPO-FR01012_6001"].altitude is None


def test_files_are_listed_year_by_year_and_joined() -> None:
    asked: list[dict[str, object]] = []

    def read(request: urllib.request.Request | str) -> bytes:
        assert isinstance(request, urllib.request.Request)
        assert request.full_url == f"{stations.DOWNLOAD_API}/ParquetFile/urls"
        assert isinstance(request.data, bytes)
        body = json.loads(request.data)
        asked.append(body)
        if body["dateTimeStart"].startswith("2013"):
            return listing("https://files/b.parquet", "https://files/a.parquet")
        return listing("https://files/a.parquet", "https://files/c.parquet", "")

    urls = stations.list_files("FR", Pollutant.NITROGEN_DIOXIDE, [2013, 2014], read=read)

    assert urls == ["https://files/a.parquet", "https://files/b.parquet", "https://files/c.parquet"]
    assert [(a["dateTimeStart"], a["dateTimeEnd"]) for a in asked] == [
        ("2013-01-01T00:00:00Z", "2014-01-01T00:00:00Z"),
        ("2014-01-01T00:00:00Z", "2015-01-01T00:00:00Z"),
    ]
    assert asked[0]["pollutants"] == ["NO2"]
    assert asked[0]["countries"] == ["FR"]
    assert asked[0]["dataset"] == stations.VALIDATED


def test_an_unexpected_list_is_refused() -> None:
    with pytest.raises(ValueError, match="unexpected list"):
        stations.list_files("FR", Pollutant.PM2_5, [2013], read=lambda _: b"<html>")


def test_build_writes_each_described_point_up_to_the_last_year(tmp_path: Path) -> None:
    files = {
        "https://files/SPO-FR01011_6001.parquet": station_file(
            "SPO-FR01011_6001",
            hours(datetime(2021, 12, 31, 1), [12.0] * 24)
            + hours(datetime(2022, 1, 1, 1), [9.0] * 24),
        ),
        "https://files/SPO-FR01012_6001.parquet": station_file(
            "SPO-FR01012_6001", hours(datetime(2021, 3, 1, 1), [30.0] * 20)
        ),
        "https://files/SPO-FR09999_6001.parquet": station_file(
            "SPO-FR09999_6001", hours(datetime(2021, 3, 1, 1), [30.0] * 20)
        ),
        "https://files/SPO-FR01013_6001.parquet": station_file(
            "SPO-FR01013_6001", hours(datetime(2022, 3, 1, 1), [30.0] * 20)
        ),
    }
    points = metadata(
        f"FR\tFR01011\tSPO-FR01011_6001\t{POLLUTANT_URL}6001\t6.18\t49.12\t192\tbackground\turban",
        f"FR\tFR01012\tSPO-FR01012_6001\t{POLLUTANT_URL}6001\t2.35\t48.86\t35\ttraffic\turban",
        f"FR\tFR01013\tSPO-FR01013_6001\t{POLLUTANT_URL}6001\t4.85\t45.75\t170\tbackground\turban",
    )

    def read(request: urllib.request.Request | str) -> bytes:
        if isinstance(request, urllib.request.Request):
            return listing(*files)
        if request == stations.METADATA_URL:
            return points
        return files[request]

    path = stations.build("FR", Pollutant.PM2_5, tmp_path, last_year=2021, read=read, downloads=2)

    assert path.name == "stations-fr-pm2_5.parquet"
    table = pq.read_table(path)
    assert table.column_names == ["sampling_point", "day", "mean", "hours"]
    assert table["sampling_point"].to_pylist() == ["SPO-FR01011_6001", "SPO-FR01012_6001"]
    assert table["day"].cast(pa.string()).to_pylist() == ["2021-12-31", "2021-03-01"]
    assert table["mean"].to_pylist() == [12.0, 30.0]
    assert table["hours"].to_pylist() == [24, 20]
    described = read_metadata(path)
    assert described["pollutant"] == "pm2_5"
    assert described["files"] == 4
    assert described["min_hours"] == 18
    assert [p["station"] for p in described["points"]] == ["FR01011", "FR01012"]
    assert described["points"][1]["type"] == "traffic"


def test_an_empty_table_still_has_its_columns() -> None:
    table = stations.to_table([], {"points": []})

    assert table.num_rows == 0
    assert table.column_names == ["sampling_point", "day", "mean", "hours"]
