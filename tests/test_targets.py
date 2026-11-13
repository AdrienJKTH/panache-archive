import io
import json
import urllib.request
import zipfile
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import openpyxl
import pyarrow.parquet as pq
import pytest

from panache_archive import targets
from panache_archive.build import read_metadata

DAY = datetime(2017, 1, 10)


def chen_text(header: str, rows: Sequence[tuple[datetime, Sequence[float | None]]]) -> str:
    """A text file as Chen et al. wrote them: tabs, empty cells, a stray `r` after each time."""
    lines = [header + "\tDateTime_UTCr"]
    for when, values in rows:
        cells = ["" if v is None else str(v) for v in values]
        lines.append("\t".join([*cells, when.strftime("%Y-%m-%d %H:%M:%S") + "r"]))
    return "\n".join(lines) + "\n"


def half_hours(start: datetime, count: int) -> list[datetime]:
    return [start + timedelta(minutes=30 * i) for i in range(count)]


def chen_zip() -> bytes:
    full_day = half_hours(DAY, 48)
    short_day = half_hours(DAY + timedelta(days=1), 30)  # 15 hours: too few
    pmf = chen_text(
        "HOA\tBBOA\tLO_OOA\tMO_OOA",
        [(t, [1.0, 2.0, 0.5, 1.5]) for t in full_day]
        + [(t, [1.0, None, 0.5, 1.5]) for t in short_day],
    )
    external = chen_text("Org\tBC", [(t, [4.0, 0.8]) for t in full_day])
    unknown = chen_text("HOA", [(t, [1.0]) for t in full_day])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data/PMF_data/par_PMF++.txt", pmf)
        archive.writestr("data/External_data/par++.txt", external)
        archive.writestr("data/PMF_data/xyz_PMF++.txt", unknown)
        archive.writestr("__MACOSX/data/PMF_data/._par_PMF++.txt", "junk")
        archive.writestr("data/Read_me_files/READ_ME_par.docx", b"PK")
    return buffer.getvalue()


def workbook(header: list[str], rows: list[list[object]], coordinates: str = "48.7, 2.16") -> bytes:
    book = openpyxl.Workbook()
    meta = book.active
    assert meta is not None
    meta.title = "Metadata"
    meta.append(["MEASUREMENT SITE:", "SIRTA, Paris, France"])
    meta.append(["TYPE OF STATION:", "Suburban"])
    meta.append(["COORDINATES:", coordinates])
    meta.append(["PI - CONTACT:", "someone (someone@example.org)"])
    meta.append(["Instrumet:", "AE33 Magee Scientific"])
    data = book.create_sheet("eBC-Abs")
    data.append(["UTC", *(["μg m-3"] * (len(header) - 1))])
    data.append(header)
    for row in rows:
        data.append(row)
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def ri_zip() -> bytes:
    hours = [DAY + timedelta(hours=h) for h in range(24)]
    ae33 = workbook(
        ["Date-Time", "BC 880", "Abs 370", "%BCff", "%BCwb", "BCff", "BCwb"],
        [[t, 1.0, 9.0, 60.0, 40.0, 0.6, 0.4] for t in hours]
        + [[t.strftime("%Y-%m-%d %H:%M:%S"), -999, 9.0, 0, 0, None, "x"] for t in hours[:2]],
    )
    maap = workbook(
        ["Date-Time", "eBC Nominal MAC", "Abs 637"],
        [[t + timedelta(days=1), 2.0, 13.0] for t in hours],
        coordinates="52.0, 4.4",
    )
    lost = workbook(["Date-Time", "BC 880"], [[t, 1.0] for t in hours], coordinates="unknown")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("eBC/AE33/Paris_SIRTA_AE33_2014-2020.xlsx", ae33)
        archive.writestr("eBC/MAAP/Rotterdam(RDM)_MAAP_2007-2021.xlsx", maap)
        archive.writestr("eBC/MAAP/Nowhere_MAAP_2010.xlsx", lost)
        archive.writestr("eBC/AE33/~$Paris_SIRTA_AE33_2014-2020.xlsx", b"lock")
        archive.writestr("eBC/READ_ME.docx", b"PK")
    return buffer.getvalue()


def test_daily_keeps_days_with_18_valid_hours() -> None:
    times = np.array(half_hours(DAY, 48) + half_hours(DAY + timedelta(days=1), 34), "M8[s]")
    values = np.r_[np.full(48, 2.0), np.full(34, 3.0)]
    values[0] = np.nan
    found = targets.daily(times, values)
    assert found.days.tolist() == [DAY.date()]
    assert found.means.tolist() == [2.0]
    assert found.hours.tolist() == [24]


def test_daily_counts_coarse_samples_for_their_step() -> None:
    times = np.array([DAY + timedelta(hours=4 * i) for i in range(6 + 4)], "M8[s]")
    values = np.r_[np.full(6, 1.0), np.full(4, 2.0)]  # the second day: 16 hours
    found = targets.daily(times, values)
    assert found.days.tolist() == [DAY.date()]
    assert found.hours.tolist() == [24]


def test_daily_counts_hourly_times_a_minute_early() -> None:
    minutes = [60 * h - (1 if h % 3 == 2 else 0) for h in range(24)]
    times = np.array([DAY + timedelta(minutes=m) for m in minutes] + [DAY], "M8[s]")
    values = np.r_[np.full(24, 2.0), 9.0]  # the last time is held twice: its first value counts
    found = targets.daily(times, values)
    assert found.hours.tolist() == [24]
    assert found.means.tolist() == [2.0]


def test_daily_drops_fill_values() -> None:
    times = np.array([DAY + timedelta(hours=h) for h in range(24)], "M8[s]")
    values = np.full(24, 5.0)
    values[:7] = -9999.0
    assert targets.daily(times, values).days.size == 0


def test_chen_series_reads_factors_and_species() -> None:
    sites, series = targets.chen_series(chen_zip())
    assert [s.key for s in sites] == ["par"]
    (site,) = sites
    assert (site.lat, site.lon, site.type, site.instrument) == (48.70889, 2.14267, "urban", "ACSM")
    assert site.time_step_minutes == 30.0
    assert site.first_day == site.last_day == "2017-01-10"
    assert set(site.variables) == {"hoa", "bboa", "lo_ooa", "mo_ooa", "ooa", "acsm_org", "acsm_bc"}
    means = {v: (d.days.tolist(), d.means.tolist()) for _, v, d in series}
    assert means["ooa"] == ([DAY.date()], [2.0])
    assert means["acsm_org"] == ([DAY.date()], [4.0])


def test_read_chen_text_names_unusual_factors() -> None:
    times, columns = targets.read_chen_text(chen_text("HOA\t'58-OA'", [(DAY, [1.0, 2.0])]))
    assert times.tolist() == [DAY]
    assert sorted(columns) == ["hoa", "oa_58"]


def test_ri_urbans_series_reads_the_split_where_it_exists() -> None:
    sites, series = targets.ri_urbans_series(ri_zip())
    assert [s.key for s in sites] == ["Paris_SIRTA_AE33_2014-2020", "Rotterdam(RDM)_MAAP_2007-2021"]
    paris, rotterdam = sites
    assert (paris.country, paris.type, paris.lat, paris.lon) == ("FR", "Suburban", 48.7, 2.16)
    assert paris.instrument == "AE33 Magee Scientific"
    assert paris.variables == ("ebc", "ebc_residential", "ebc_traffic")
    assert rotterdam.variables == ("ebc",)
    assert rotterdam.country == "NL"
    assert paris.time_step_minutes == 60.0
    means = {(s, v): d.means.tolist() for s, v, d in series}
    assert means["Paris_SIRTA_AE33_2014-2020", "ebc_traffic"] == pytest.approx([0.6])
    assert means["Paris_SIRTA_AE33_2014-2020", "ebc"] == pytest.approx([1.0])
    assert means["Rotterdam(RDM)_MAAP_2007-2021", "ebc"] == pytest.approx([2.0])


def test_build_writes_one_asset_per_record(tmp_path: Path) -> None:
    files = {
        targets.CHEN_RECORD: ("data_for_submission.zip", chen_zip()),
        targets.RI_URBANS_RECORD: ("eBC data-RI-URBANS.zip", ri_zip()),
    }

    def read(request: urllib.request.Request | str) -> bytes:
        url = request.full_url if isinstance(request, urllib.request.Request) else request
        for record, (name, content) in files.items():
            if url == targets.ZENODO_API + record:
                link = f"https://zenodo.org/files/{record}/{name}"
                about = {
                    "doi": f"10.5281/zenodo.{record}",
                    "metadata": {"version": "2nd"},
                    "files": [{"key": name, "checksum": "md5:0", "links": {"self": link}}],
                }
                return json.dumps(about).encode()
            if url.endswith(f"{record}/{name}"):
                return content
        raise AssertionError(url)

    paths = targets.build(tmp_path, read=read)
    assert [p.name for p in paths] == [targets.CHEN_ASSET, targets.RI_URBANS_ASSET]
    chen = pq.read_table(paths[0])
    assert chen.column_names == ["site", "variable", "day", "mean", "hours"]
    assert set(chen["variable"].to_pylist()) >= {"bboa", "hoa", "ooa"}
    metadata = read_metadata(paths[1])
    assert metadata["license"] == "CC BY 4.0"
    assert "101036245" in metadata["attribution"]
    assert "someone@example.org" not in json.dumps(metadata)
    assert len(metadata["sites"]) == 2
    described = targets.describe(paths[1])
    assert "| Paris_SIRTA_AE33_2014-2020 | FR | Suburban |" in described
    assert "ebc 1, ebc_residential 1, ebc_traffic 1" in described


def test_to_table_of_nothing() -> None:
    assert targets.to_table([], {}).num_rows == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("51.52253, -0.154611", (51.52253, -0.154611)),
        ("37°58'23.3\"N, 23°43'08.9\"E", (37.973139, 23.719139)),
        ("40.456 N, 3.726 W", (40.456, -3.726)),
        ("N/A", None),
        ("95, 10", None),
        ("37, 58", None),
        ("39.51\u00b0, \u22120.42\u00b0", (39.51, -0.42)),
        (
            "60\u00b017\u201930.2\u2019\u2019, 25\u00b006\u201946.3\u2019\u2019",
            (60.291722, 25.112861),
        ),
        ("45\u00ba28\u00b444\u00b4\u00b4N, 9\u00ba14\u00b407\u00b4\u00b4E", (45.478889, 9.235278)),
        ("43°18'19.1\"N, 43°18'19.1\"N", None),
    ],
)
def test_coordinates(text: str, expected: tuple[float, float] | None) -> None:
    assert targets._coordinates(text) == expected


@pytest.mark.parametrize(
    ("key", "place"),
    [
        ("SMEAR II_Hyytiälä_AE33_2018-2021", "SMEAR II"),
        ("Rotterdam(RZW)_MAAP_2010-2021", "Rotterdam"),
        ("Lugano_MAAP_2007-2021", "Lugano"),
    ],
)
def test_first_word(key: str, place: str) -> None:
    assert targets._first_word(key) == place


@pytest.mark.parametrize(("cell", "expected"), [("1,25", 1.25), ("1.25", 1.25), ("n/a", None)])
def test_number_reads_a_decimal_comma(cell: str, expected: float | None) -> None:
    found = targets._number(cell)
    assert (None if np.isnan(found) else found) == expected


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        (datetime(2017, 1, 31, 13), "2017-01-31T13:00:00"),
        (42736.5, "2017-01-01T12:00:00"),
        ("2017-01-31 13:00:00", "2017-01-31T13:00:00"),
        ("31/01/2017 13:00", "2017-01-31T13:00:00"),
        ("31/01/2017", "2017-01-31T00:00:00"),
        ("soon", None),
        (None, None),
        (True, None),
    ],
)
def test_time(cell: object, expected: str | None) -> None:
    found = targets._time(cell)
    assert (None if found is None else str(found)) == expected
