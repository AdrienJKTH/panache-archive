import gzip
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from panache_archive import cli, live_compare, live_stations
from panache_archive.build import METADATA_KEY
from panache_archive.live_compare import LiveSeries, Pairs
from panache_archive.live_stations import Screening
from panache_archive.stations import PointDays, Pollutant, Read, to_table

HEADER = (
    '"location_id","sensors_id","location","datetime","lat","lon","parameter","units","value"\n'
)


def openaq_day(
    day: str, values: list[float], parameter: str = "pm25", units: str = "µg/m³"
) -> bytes:
    """A day's file of OpenAQ's archive: hours stamped by their end, at UTC+1."""
    lines = []
    for hour, value in enumerate(values):
        end = np.datetime64(f"{day}T00", "h") + np.timedelta64(hour + 2, "h")  # +1 h zone, +1 end
        lines.append(
            f'7,1,"Site-7","{end!s}:00:00+01:00","49.6","6.1","{parameter}","{units}","{value}"'
        )
    return gzip.compress((HEADER + "\n".join(lines) + "\n").encode())


def validated_asset(folder: Path, country: str, pollutant: Pollutant, values: list[float]) -> Path:
    days = np.datetime64("2025-03-01") + np.arange(len(values))
    found = [PointDays("SPO-1", days, np.array(values), np.full(len(values), 24))]
    points = [
        {
            "sampling_point": "SPO-1",
            "station": "S1",
            "lat": 49.6000,
            "lon": 6.1003,  # ~22 m east of OpenAQ's location
            "altitude": None,
            "type": "background",
            "area": "rural-regional",
        }
    ]
    path = folder / f"stations-{country.lower()}-{pollutant}.parquet"
    meta = {"country": country, "points": points}
    pq.write_table(to_table(found, meta), path)
    return path


def fake_archive(files: dict[str, bytes]) -> Read:
    def read(request: object) -> bytes:
        url = str(request)
        if "list-type=2" in url:
            if "continuation-token" not in url:
                keys = "".join(f"<Key>{k}</Key>" for k in list(files)[:1])
                return f"<r>{keys}<NextContinuationToken>a/b</NextContinuationToken></r>".encode()
            return "".join(f"<Key>{k}</Key>" for k in list(files)[1:]).encode()
        return files[url.removeprefix(f"{live_compare.ARCHIVE}/")]

    return read


def test_openaq_hours_become_utc_starts_by_pollutant() -> None:
    files = {
        "k/day1.csv.gz": openaq_day("2025-03-01", [1.0, 2.0]),
        "k/day2.csv.gz": openaq_day("2025-03-02", [3.0], parameter="o3"),
    }

    found = live_compare.read_location(7, 2025, read=fake_archive(files))

    assert [s.pollutant for s in found] == [Pollutant.OZONE, Pollutant.PM2_5]
    assert found[1].hours.astype(str).tolist() == ["2025-03-01T00:00:00", "2025-03-01T01:00:00"]
    assert found[1].values.tolist() == [1.0, 2.0]
    assert (found[0].lat, found[0].lon) == (49.6, 6.1)


def test_other_units_parameters_and_empty_values_are_left_out() -> None:
    rows = [
        {"parameter": "pm25", "units": "ppm", "value": "1", "datetime": "2025-01-01T01:00:00Z"},
        {"parameter": "no2", "units": "µg/m³", "value": "1", "datetime": "2025-01-01T01:00:00Z"},
        {"parameter": "pm10", "units": "µg/m³", "value": "", "datetime": "2025-01-01T01:00:00Z"},
    ]

    assert live_compare.series_of_rows(7, rows) == []


def test_a_utc_stamp_and_a_western_offset_are_read() -> None:
    assert str(live_compare._utc("2025-01-01T01:00:00Z")) == "2025-01-01T01:00:00"
    assert str(live_compare._utc("2025-01-01T01:00:00-01:30")) == "2025-01-01T02:30:00"


def test_classes_follow_panache_s_station_class() -> None:
    assert live_compare.station_class("traffic", "urban") == "traffic"
    assert live_compare.station_class("industrial", "rural") == "industrial"
    assert live_compare.station_class("background", "rural-remote") == "rural"
    assert live_compare.station_class("background", "suburban") == "suburban"
    assert live_compare.station_class("background", "urban") == "urban"


def test_a_location_matches_the_nearest_point_within_100_m(tmp_path: Path) -> None:
    validated = live_compare.read_validated(
        [validated_asset(tmp_path, "LU", Pollutant.PM2_5, [1.0])]
    )

    assert live_compare.nearest(validated, 49.6, 6.1) == 0
    assert live_compare.nearest(validated, 49.6, 6.11) is None
    empty = live_compare.Validated(
        np.array([]), np.array([]), (), np.array([], np.int64), np.array([]), np.array([])
    )
    assert live_compare.nearest(empty, 49.6, 6.1) is None


def test_validated_assets_need_dictionary_encoded_points(tmp_path: Path) -> None:
    import pyarrow as pa

    table = pa.table({"sampling_point": ["SPO-1"], "day": [0], "mean": [1.0]})
    path = tmp_path / "x.parquet"
    pq.write_table(table.replace_schema_metadata({METADATA_KEY: '{"points": []}'}), path)

    with pytest.raises(ValueError, match="dictionary-encoded"):
        live_compare.read_validated([path])


def series(values: list[float], day: str = "2025-03-01") -> LiveSeries:
    hours = np.datetime64(f"{day}T00", "s") + np.arange(len(values)) * np.timedelta64(1, "h")
    return LiveSeries(7, Pollutant.PM2_5, 49.6, 6.1, hours, np.array(values))


def test_each_validated_day_is_paired_with_the_live_day(tmp_path: Path) -> None:
    validated = live_compare.read_validated(
        [validated_asset(tmp_path, "LU", Pollutant.PM2_5, [10.0, 20.0])]
    )
    live = series([9.0, 11.0] * 12)  # 1 March only
    far = LiveSeries(8, Pollutant.PM2_5, 40.0, 6.1, live.hours, live.values)

    found = live_compare.pair(
        Pollutant.PM2_5, [live, live, far], validated, Screening(Pollutant.PM2_5), 2025
    )

    assert found.classes == ["rural", "rural"]
    assert found.validated == [10.0, 20.0]
    assert found.live[0] == 10.0
    assert math.isnan(found.live[1])


def test_scores_give_r2_rmse_bias_and_the_log_spread() -> None:
    pairs = Pairs(
        Pollutant.PM2_5,
        ["urban", "urban", "urban", "traffic"],
        [10.0, 20.0, 30.0, 5.0],
        [12.0, 22.0, 32.0, math.nan],
    )

    rows = live_compare.scores(pairs)

    assert [r["class"] for r in rows] == ["all", "urban", "traffic"]
    assert rows[0]["missing_live"] == 0.25
    assert rows[1] == {
        "pollutant": "pm2_5",
        "class": "urban",
        "days": 3,
        "missing_live": 0.0,
        "r2": 0.94,
        "rmse": 2.0,
        "bias": 2.0,
        "sd_log": rows[1]["sd_log"],
    }
    assert 0 < rows[1]["sd_log"] < 0.1
    assert "r2" not in rows[2]


def test_ozone_has_no_log_spread() -> None:
    rows = live_compare.scores(Pairs(Pollutant.OZONE, ["urban"] * 2, [1.0, 1.0], [1.0, 2.0]))

    assert rows[0]["r2"] is None
    assert "sd_log" not in rows[0]


def test_the_command_scores_every_pollutant_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "validated"
    folder.mkdir()
    validated_asset(folder, "LU", Pollutant.PM2_5, [10.0])
    (tmp_path / "ids.txt").write_text("7,")
    files = {"k/d.csv.gz": openaq_day("2025-03-01", [9.0, 11.0] * 12)}
    out = tmp_path / "out.json"

    result = live_compare.compare(
        [7],
        2025,
        {Pollutant.PM2_5: live_compare.read_validated([folder / "stations-lu-pm2_5.parquet"])},
        read=fake_archive(files),
        downloads=1,
    )

    assert result["series"] == {"pm2_5": 1}
    assert result["scores"][0]["days"] == 1
    assert "| pm2_5 | all | 1 |" in live_compare.markdown(result)

    def fake_compare(
        locations: list[int], _year: int, validated: dict[Pollutant, object]
    ) -> dict[str, object]:
        assert locations == [7]
        assert list(validated) == [Pollutant.PM2_5]
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(live_compare, "compare", fake_compare)
        code = cli.main(
            [
                "live-compare",
                "--year",
                "2025",
                "--locations",
                str(tmp_path / "ids.txt"),
                "--validated",
                str(folder),
                "--out",
                str(out),
            ]
        )
    assert code == 0
    assert json.loads(out.read_text())["year"] == 2025
    assert "pm2_5" in capsys.readouterr().out


def test_the_live_command_builds_yesterday_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_build(
        pollutant: Pollutant, first: object, last: object, out: Path, *, countries: object
    ) -> list[Path]:
        calls.append((pollutant, first, last, countries))
        return [out / "a.parquet"]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(live_stations, "build", fake_build)
        cli.main(["live-stations", "--pollutant", "pm10", "--out", str(tmp_path)])
        cli.main(
            [
                "live-stations",
                "--pollutant",
                "ozone",
                "--first-day",
                "2026-11-01",
                "--last-day",
                "2026-11-07",
                "--countries",
                "FR,DE",
                "--out",
                str(tmp_path),
            ]
        )

    from datetime import date, timedelta

    yesterday = date.today() - timedelta(days=1)
    assert calls == [
        (Pollutant.PM10, yesterday, yesterday, None),
        (Pollutant.OZONE, date(2026, 11, 1), date(2026, 11, 7), ["FR", "DE"]),
    ]
    assert "a.parquet" in capsys.readouterr().out
