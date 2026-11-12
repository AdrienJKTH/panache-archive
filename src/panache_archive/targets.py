"""Measured sources of carbonaceous aerosol at European sites, the targets Panache's feature 4
learns from in its public version (Panache's D87): wood smoke or traffic, from open measurements.

Two records, both CC BY 4.0 on Zenodo, become one Parquet asset each:

- **Chen et al. 2022** (doi:10.5281/zenodo.6672710): the organic aerosol factors of a harmonised
  rolling PMF at 22 sites, 21 ACSM and 1 AMS, 2013-2019, one or two years a site, every ~30
  minutes: HOA (hydrocarbon-like, traffic), BBOA (biomass burning), the oxygenated LO-OOA and
  MO-OOA and their sum OOA, and factors of a few sites (cooking, coal, peat, sea salt, ship and
  industry...), with the ACSM's own inorganics, total organics and black carbon. The sites'
  positions are the record's `Site_descriptions.docx`, copied into `CHEN_SITES`.
- **RI-URBANS, Savadkoohi et al. 2023** (doi:10.5281/zenodo.7982201): equivalent black carbon
  (eBC) at 56 instrument series of ~45 European stations, 2006-2022, hourly. The 26 AE33
  aethalometer series also hold the aethalometer model's split (Sandradewi et al. 2008, 470 and
  950 nm, Ångström exponents 1 and 2): eBC from traffic (`BCff`, fossil fuel) and from
  residential and commercial combustion (`BCwb`, wood burning). The MAAP, AE31 and AE22 series hold
  total eBC only. Each workbook's `Metadata` sheet gives the station's type, position and
  instrument; its contacts are left out of the asset.

Each series' UTC days become daily means by the rule the archive's stations follow (Panache's
D69): a day needs at least 18 valid hours. A sample counts for the hour it starts in; a series
sampled less often than hourly counts each valid sample for its whole time step.
"""

import io
import json
import logging
import re
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from panache_archive import web
from panache_archive.build import METADATA_KEY

FloatArray = NDArray[np.float64]
Read = Callable[[urllib.request.Request | str], bytes]

ZENODO_API: Final = "https://zenodo.org/api/records/"
CHEN_RECORD: Final = "6672710"
RI_URBANS_RECORD: Final = "7982201"
CHEN_ASSET: Final = "chen-2022-oa-factors.parquet"
RI_URBANS_ASSET: Final = "ri-urbans-ebc.parquet"
MIN_HOURS: Final = 18
"""Valid hours a UTC day needs, as the stations' and CAMS's daily means (Panache's D69)."""
VALUE_DECIMALS: Final = 4
MISSING_BELOW: Final = -900.0
"""Values at or below this are a workbook's fill value (-999, -9999), not a measurement."""
USER_AGENT: Final = "panache-archive (+https://github.com/AdrienJKTH/panache-archive)"
CHEN_ATTRIBUTION: Final = (
    "Chen, G. et al. (2022), European aerosol phenomenology - 8: Harmonised source apportionment "
    "of organic aerosol using 22 year-long ACSM/AMS datasets, Environ. Int. 166, 107325, "
    "doi:10.1016/j.envint.2022.107325; data doi:10.5281/zenodo.6672710, CC BY 4.0."
)
RI_URBANS_ATTRIBUTION: Final = (
    "Savadkoohi, M. et al. (2023), The variability of mass concentrations and source "
    "apportionment analysis of equivalent black carbon across urban Europe, Environ. Int. 178, "
    "108081, doi:10.1016/j.envint.2023.108081; data doi:10.5281/zenodo.7982201, CC BY 4.0. "
    "RI-URBANS has received funding from the European Union's Horizon 2020 research and "
    "innovation programme under grant agreement No 101036245. Use may require contact with the "
    "data providers named in each workbook."
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Site:
    """Where a series was measured, and what it holds."""

    key: str
    name: str
    country: str
    lat: float
    lon: float
    type: str
    instrument: str
    variables: tuple[str, ...]
    time_step_minutes: float
    first_day: str
    last_day: str
    source_file: str


@dataclass(frozen=True, slots=True)
class Daily:
    """One series' daily means: `days`, `means` and the valid `hours` behind each."""

    days: NDArray[np.datetime64]
    means: FloatArray
    hours: NDArray[np.int64]


CHEN_SITES: Final[Mapping[str, tuple[str, str, float, float, str]]] = {
    # acronym: (name, country, latitude, longitude, type), Site_descriptions.docx, tables 1 and 2.
    "bir": ("Birkenes", "NO", 58.38333, 8.25, "non-urban"),
    "casp": ("Carnsore Point", "IE", 52.1706, -6.3556, "non-urban"),
    "hyy": ("Hyytiälä", "FI", 61.85, 24.28333, "non-urban"),
    "hoh": ("Hohenpeissenberg", "DE", 47.80139, 11.00972, "non-urban"),
    "puy": ("Puy de Dôme", "FR", 45.77222, 2.96583, "non-urban"),
    "cao": ("Cyprus Atmospheric Observatory", "CY", 35.04, 33.06, "non-urban"),
    "kos": ("Košetice", "CZ", 49.5733, 15.081, "non-urban"),
    "mag": ("Magadino", "CH", 46.1464, 8.8519, "non-urban"),
    "mel": ("Melpitz", "DE", 51.54, 12.93, "non-urban"),
    "dub": ("Dublin", "IE", 53.345, -6.25417, "urban"),
    "tar": ("Tartu", "EE", 58.37069, 26.73575, "urban"),
    "hel": ("Helsinki", "FI", 60.19639, 24.95194, "urban"),
    "athd": ("Athens Demokritos", "GR", 37.99528, 23.81583, "urban"),
    "zur": ("Zürich", "CH", 47.36667, 8.55, "urban"),
    "bar": ("Barcelona", "ES", 41.38722, 2.11573, "urban"),
    # The table gives 0.21° E; its link is North Kensington (KC1), at 0.21° W.
    "lon": ("London North Kensington", "GB", 51.52, -0.21, "urban"),
    "par": ("Paris SIRTA", "FR", 48.70889, 2.14267, "urban"),
    "mar": ("Marseille", "FR", 43.30531, 5.39469, "urban"),
    "athn": ("Athens NOA", "GR", 37.9747, 23.7164, "urban"),
    "lil": ("Lille", "FR", 50.61105, 3.14037, "urban"),
    "buc": ("Bucharest", "RO", 44.35, 26.1, "urban"),
    "kra": ("Kraków", "PL", 50.06556, 19.91556, "urban"),
}
CHEN_INSTRUMENT: Final = "ACSM"
CHEN_AMS_SITES: Final = frozenset({"bir"})
"""Birkenes' organics come from an AMS at a coarser step (Site_descriptions.docx)."""
CHEN_TIME: Final = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
CHEN_FILE: Final = re.compile(r"(?:PMF|External)_data/(?P<site>[a-z]+)(?:_PMF)?\+\+\.txt$")

RI_COLUMNS: Final[Mapping[str, tuple[str, ...]]] = {
    # variable: the workbook headers that may hold it, in order of preference
    "ebc": ("BC 880", "eBC Nominal MAC"),
    "ebc_traffic": ("BCff",),
    "ebc_residential": ("BCwb",),
}
RI_DATA_SHEET: Final = "eBC-Abs"
RI_METADATA_SHEET: Final = "Metadata"
RI_COUNTRY: Final[Mapping[str, str]] = {
    "Athens": "GR",
    "Barcelona": "ES",
    "BCN": "ES",
    "Bern": "CH",
    "Birmingham": "GB",
    "Bucharest": "RO",
    "DDN": "DE",
    "DDW": "DE",
    "Granada": "ES",
    "Helsinki": "FI",
    "Ispra": "IT",
    "Leipzig": "DE",
    "Lille": "FR",
    "London": "GB",
    "Lugano": "CH",
    "Madrid": "ES",
    "Marseille": "FR",
    "Milan": "IT",
    "NLD": "NL",
    "Paris": "FR",
    "Rotterdam": "NL",
    "SMEAR II": "FI",
    "Stockholm": "SE",
    "Valencia": "ES",
    "Zurich": "CH",
}
"""Each workbook name's first word (DDN and DDW: Dresden), to its country."""


def daily(times: NDArray[np.datetime64], values: FloatArray) -> Daily:
    """Each UTC day's mean of the valid `values`, for days with at least `MIN_HOURS` valid hours.

    The time step is the series' median gap. Up to an hour, a day's valid hours are the distinct
    hours its valid samples start in; beyond, each valid sample counts for the whole step."""
    seconds = times.astype("datetime64[s]").astype(np.int64)
    order = np.argsort(seconds, kind="stable")
    seconds, values = seconds[order], values[order]
    gaps = np.diff(np.unique(seconds))
    step = float(np.median(gaps)) if gaps.size else 3600.0
    valid = ~np.isnan(values) & (values > MISSING_BELOW)
    seconds, values = seconds[valid], values[valid]
    day = seconds // 86_400
    days, index = np.unique(day, return_inverse=True)
    sums = np.bincount(index, weights=values, minlength=days.size)
    counts = np.bincount(index, minlength=days.size)
    if step <= 3600:
        hour = seconds // 3600
        _, first = np.unique(hour, return_index=True)
        hours = np.bincount(index[first], minlength=days.size)
    else:
        hours = np.minimum(np.round(counts * step / 3600), 24).astype(np.int64)
    full = hours >= MIN_HOURS
    return Daily(
        days=days[full].astype("datetime64[D]"),
        means=sums[full] / counts[full],
        hours=hours[full].astype(np.int64),
    )


def time_step_minutes(times: NDArray[np.datetime64]) -> float:
    """A series' median gap between distinct times, in minutes."""
    seconds = np.unique(times.astype("datetime64[s]").astype(np.int64))
    return round(float(np.median(np.diff(seconds))) / 60, 1) if seconds.size > 1 else float("nan")


def read_chen_text(text: str) -> tuple[NDArray[np.datetime64], dict[str, FloatArray]]:
    """A Chen et al. text file's times (UTC) and columns: tab-separated, a header whose last
    column is the time, empty cells missing. Each line's time is followed by a stray `r`."""
    lines = [line for line in text.splitlines() if line.strip()]
    header = lines[0].rstrip("r").split("\t")
    names = [_variable(name) for name in header[:-1]]
    times: list[str] = []
    rows: list[list[float]] = []
    for line in lines[1:]:
        cells = line.split("\t")
        found = CHEN_TIME.search(cells[-1])
        if found is None:
            continue
        times.append(found.group(1).replace(" ", "T"))
        rows.append([_number(cell) for cell in cells[: len(names)]])
    table = np.array(rows, dtype=np.float64).reshape(len(rows), len(names))
    columns = {name: table[:, i] for i, name in enumerate(names)}
    if "ooa" not in columns and {"lo_ooa", "mo_ooa"} <= columns.keys():
        columns["ooa"] = columns["lo_ooa"] + columns["mo_ooa"]
    return np.array(times, dtype="datetime64[s]"), columns


def chen_series(archive: bytes) -> tuple[list[Site], list[tuple[str, str, Daily]]]:
    """Every site's daily means of its factors (PMF_data) and of its ACSM's species
    (External_data), and the sites."""
    found: dict[str, list[tuple[str, Daily]]] = {}
    steps: dict[str, list[NDArray[np.datetime64]]] = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        for name in sorted(zipped.namelist()):
            matched = CHEN_FILE.search(name)
            if matched is None or "__MACOSX" in name:
                continue
            site = matched["site"]
            external = "External_data" in name
            times, columns = read_chen_text(zipped.read(name).decode("utf-8", errors="replace"))
            steps.setdefault(site, []).append(times)
            for variable, values in columns.items():
                label = f"acsm_{variable}" if external else variable
                found.setdefault(site, []).append((label, daily(times, values)))
    sites: list[Site] = []
    series: list[tuple[str, str, Daily]] = []
    for key in sorted(found):
        if key not in CHEN_SITES:
            logger.warning("Chen et al.: no position for %s, left out", key)
            continue
        name, country, lat, lon, kind = CHEN_SITES[key]
        kept = [(v, d) for v, d in found[key] if d.days.size]
        series.extend((key, variable, days) for variable, days in kept)
        all_days = np.concatenate([d.days for _, d in kept]) if kept else np.array([], "M8[D]")
        sites.append(
            Site(
                key=key,
                name=name,
                country=country,
                lat=lat,
                lon=lon,
                type=kind,
                instrument="AMS" if key in CHEN_AMS_SITES else CHEN_INSTRUMENT,
                variables=tuple(sorted({v for v, _ in kept})),
                time_step_minutes=time_step_minutes(np.concatenate(steps[key])),
                first_day=str(all_days.min()) if all_days.size else "",
                last_day=str(all_days.max()) if all_days.size else "",
                source_file=f"{key}_PMF++.txt",
            )
        )
    return sites, series


def ri_urbans_series(archive: bytes) -> tuple[list[Site], list[tuple[str, str, Daily]]]:
    """Every workbook's daily means of eBC and, where it holds them, of eBC from traffic and from
    residential combustion, and its site."""
    import openpyxl  # only this target needs it

    sites: list[Site] = []
    series: list[tuple[str, str, Daily]] = []
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        for name in sorted(zipped.namelist()):
            file = name.rsplit("/", 1)[-1]
            if not file.endswith(".xlsx") or file.startswith("~$") or "__MACOSX" in name:
                continue
            book = openpyxl.load_workbook(
                io.BytesIO(zipped.read(name)), read_only=True, data_only=True
            )
            metadata = _metadata(book[RI_METADATA_SHEET].iter_rows(values_only=True))
            times, columns = _ri_columns(book[RI_DATA_SHEET].iter_rows(values_only=True))
            book.close()
            key = file.removesuffix(".xlsx")
            position = _coordinates(metadata.get("COORDINATES", ""))
            if position is None:
                logger.warning("RI-URBANS: no coordinates in %s, left out", file)
                continue
            kept = [(v, daily(times, values)) for v, values in columns.items()]
            kept = [(v, d) for v, d in kept if d.days.size]
            series.extend((key, variable, days) for variable, days in kept)
            all_days = np.concatenate([d.days for _, d in kept]) if kept else np.array([], "M8[D]")
            sites.append(
                Site(
                    key=key,
                    name=metadata.get("MEASUREMENT SITE", key),
                    country=RI_COUNTRY.get(_first_word(key), ""),
                    lat=position[0],
                    lon=position[1],
                    type=metadata.get("TYPE OF STATION", ""),
                    instrument=metadata.get("Instrumet", metadata.get("Instrument", "")),
                    variables=tuple(sorted(v for v, _ in kept)),
                    time_step_minutes=time_step_minutes(times),
                    first_day=str(all_days.min()) if all_days.size else "",
                    last_day=str(all_days.max()) if all_days.size else "",
                    source_file=file,
                )
            )
            logger.info("%s: %s", file, {v: int(d.days.size) for v, d in kept})
    return sites, series


def build(out: Path, *, read: Read = web.read) -> list[Path]:
    """Both records' assets in `out`: one row per series, variable and UTC day."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for record, asset, parse, attribution in (
        (CHEN_RECORD, CHEN_ASSET, chen_series, CHEN_ATTRIBUTION),
        (RI_URBANS_RECORD, RI_URBANS_ASSET, ri_urbans_series, RI_URBANS_ATTRIBUTION),
    ):
        about = json.loads(read(_get(ZENODO_API + record)))
        (entry,) = about["files"]
        sites, series = parse(read(_get(entry["links"]["self"])))
        metadata = {
            "source": f"zenodo-{record}",
            "doi": about.get("doi", ""),
            "version": about["metadata"].get("version"),
            "license": "CC BY 4.0",
            "file": entry["key"],
            "checksum": entry["checksum"],
            "unit": "µg/m³",
            "min_hours": MIN_HOURS,
            "built": datetime.now(UTC).isoformat(timespec="seconds"),
            "attribution": attribution,
            "sites": [asdict(site) for site in sites],
        }
        path = out / asset
        pq.write_table(to_table(series, metadata), path, compression="zstd")
        logger.info("%s: %d sites, %d series", path.name, len(sites), len(series))
        paths.append(path)
    return paths


def describe(path: Path) -> str:
    """An asset's sites as a Markdown table: what they are, where, and each variable's days."""
    metadata = json.loads((pq.read_schema(path).metadata or {})[METADATA_KEY])
    table = pq.read_table(path, columns=["site", "variable"])
    sites = np.asarray(table["site"].cast(pa.string()).to_numpy(zero_copy_only=False))
    variables = np.asarray(table["variable"].cast(pa.string()).to_numpy(zero_copy_only=False))
    lines = [
        f"### {path.name}: {len(metadata['sites'])} series, {table.num_rows:,} daily means",
        "",
        "| Series | Country | Type | Instrument | Lat, lon | Step (min) | Days | By variable |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for site in metadata["sites"]:
        here = sites == site["key"]
        counts = dict(zip(*np.unique(variables[here], return_counts=True), strict=True))
        days = ", ".join(f"{v} {int(n)}" for v, n in sorted(counts.items()))
        lines.append(
            f"| {site['key']} | {site['country']} | {site['type']} | {site['instrument']} | "
            f"{site['lat']:.3f}, {site['lon']:.3f} | {site['time_step_minutes']} | "
            f"{site['first_day']} to {site['last_day']} | {days} |"
        )
    return "\n".join(lines)


def to_table(series: Sequence[tuple[str, str, Daily]], metadata: Mapping[str, Any]) -> pa.Table:
    """One row per series, variable and day, in the order given."""
    lengths = [d.days.size for _, _, d in series]

    def labels(values: Iterable[str]) -> "pa.DictionaryArray[Any, Any]":
        names = list(values)
        unique = sorted(set(names))
        codes = np.repeat(np.array([unique.index(n) for n in names], np.int32), lengths)
        return pa.DictionaryArray.from_arrays(codes, pa.array(unique, pa.string()))

    def joined(arrays: list[np.ndarray], dtype: str) -> np.ndarray:
        return np.concatenate(arrays) if arrays else np.array([], dtype)

    table = pa.table(
        {
            "site": labels(s for s, _, _ in series),
            "variable": labels(v for _, v, _ in series),
            "day": pa.array(joined([d.days for _, _, d in series], "M8[D]"), pa.date32()),
            "mean": pa.array(
                np.round(joined([d.means for _, _, d in series], "f8"), VALUE_DECIMALS).astype(
                    np.float32
                )
            ),
            "hours": pa.array(joined([d.hours for _, _, d in series], "i8").astype(np.int8)),
        }
    )
    return table.replace_schema_metadata({METADATA_KEY: json.dumps(metadata).encode()})


def _metadata(rows: Iterable[tuple[Any, ...]]) -> dict[str, str]:
    """A Metadata sheet's labels (first column, without their colon) and values (second)."""
    found: dict[str, str] = {}
    for row in rows:
        if len(row) >= 2 and row[0] is not None and row[1] is not None:
            label = str(row[0]).strip().rstrip(":").strip()
            found.setdefault(label, str(row[1]).strip())
    return found


def _ri_columns(
    rows: Iterable[tuple[Any, ...]],
) -> tuple[NDArray[np.datetime64], dict[str, FloatArray]]:
    """A data sheet's times and the `RI_COLUMNS` it holds: a row of units, a row of headers
    whose first is the time, then one row per time."""
    iterator = iter(rows)
    next(iterator)  # units
    header = [str(cell).strip() if cell is not None else "" for cell in next(iterator)]
    wanted = {
        variable: header.index(next(name for name in names if name in header))
        for variable, names in RI_COLUMNS.items()
        if any(name in header for name in names)
    }
    times: list[np.datetime64] = []
    values: list[list[float]] = []
    for row in iterator:
        when = _time(row[0] if row else None)
        if when is None:
            continue
        times.append(when)
        values.append([_number(row[i]) if i < len(row) else np.nan for i in wanted.values()])
    table = np.array(values, dtype=np.float64).reshape(len(values), len(wanted))
    return np.array(times, dtype="datetime64[s]"), {
        variable: table[:, i] for i, variable in enumerate(wanted)
    }


def _time(cell: Any) -> np.datetime64 | None:
    if isinstance(cell, datetime):
        return np.datetime64(cell.replace(tzinfo=None), "s")
    if isinstance(cell, str) and (found := CHEN_TIME.search(cell.replace("T", " "))):
        return np.datetime64(found.group(1).replace(" ", "T"), "s")
    return None


def _coordinates(text: str) -> tuple[float, float] | None:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(numbers) < 2:
        return None
    lat, lon = float(numbers[0]), float(numbers[1])
    return (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else None


def _first_word(key: str) -> str:
    """A workbook name's place: `SMEAR II`, or what precedes the first `_` or `(`."""
    if key.startswith("SMEAR II"):
        return "SMEAR II"
    return re.split(r"[_(]", key, maxsplit=1)[0]


def _variable(header: str) -> str:
    """A factor's or species' name as a column: `LO_OOA` → `lo_ooa`, `'58-OA'` → `oa_58`."""
    name = header.strip().strip("'").lower().replace("-", "_")
    return f"oa_{name.removesuffix('_oa')}" if name[:1].isdigit() else name


def _number(cell: Any) -> float:
    if cell is None or isinstance(cell, bool):
        return np.nan
    if isinstance(cell, int | float):
        return float(cell)
    try:
        return float(str(cell).strip())
    except ValueError:
        return np.nan


def _get(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
