---
license: cc-by-4.0
pretty_name: CAMS Europe daily air quality, cell by cell (Panache archive)
viewer: false
tags:
  - air-quality
  - atmosphere
  - copernicus
  - cams
  - reanalysis
  - europe
---

# CAMS Europe daily air quality, cell by cell

Daily air quality over the whole CAMS Europe domain (30 to 72° N, 25° W to 45° E, 0.1° cells) from 2013, summed up from the hourly **CAMS European air quality reanalyses**. It is built for [Panache](https://github.com/AdrienJKTH/panache-archive), a tool that explains pollution episodes at a place: was a day unusual, and why?

This dataset is a **mirror** of two GitHub releases of [AdrienJKTH/panache-archive](https://github.com/AdrienJKTH/panache-archive), where the assets are built on GitHub Actions and checked. Its files are the very same, one folder per release:
- [`cams-europe-v1/`](https://github.com/AdrienJKTH/panache-archive/releases/tag/cams-europe-v1): air quality, described below;
- [`era5-europe-v1/`](https://github.com/AdrienJKTH/panache-archive/releases/tag/era5-europe-v1): the daily weather of ERA5 over the same domain at 0.25° (169 by 281 points), one file per year such as `era5-2021.cells`, nine statistics in this order: `temperature_2m_mean` (°C, the grid's, not corrected for altitude), `relative_humidity_2m_mean` (%), `wind_speed_10m_mean` (m/s), `wind_direction_10m_dominant` (°), `pressure_msl_mean` (hPa), `precipitation_sum` (mm, the 24 hours ending in the day), `shortwave_radiation_sum` (MJ/m²), `cloud_cover_mean` (%) and `boundary_layer_height` (m). One point a frame. Built from Open-Meteo's public copy of ERA5 (DOI [10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47)); weather data by [Open-Meteo.com](https://open-meteo.com/) (CC-BY 4.0).

## What it holds

One file per variable and year, such as `cams-europe-v1/cams-pm10-2021-validated.cells`:

- **Variables** (Open-Meteo's names), in μg/m³:
  - `pm10`, `pm2_5`, `nitrogen_dioxide` and `ozone`, from 2013;
  - `carbon_monoxide` and `sulphur_dioxide`, from 2016;
  - `ammonia`, `dust` and `secondary_inorganic_aerosol`, from 2018;
  - `pm10_wildfires`, from 2019.
- **Statistics**: the UTC daily mean, null with fewer than 18 hours. Ozone also has its daily maximum 8-hour mean (EU Directive 2008/50/EC).
- **Source**: the ensemble median at the surface. The validated reanalysis where it exists; the interim one after it (`-interim` in the name), replaced once the validated stream covers the year.
- **Values**: rounded to 0.01 μg/m³.
- **Grids**: 420 by 700 cells centred on x.x5° since 2020; 421 by 701 centred on x.x0° until 2019. Each file holds its year's own grid.

## The `.cells` format

The files are laid out to give one cell's year in two small HTTP range requests, so that a place is read without downloading the files (~150 MB each). Little-endian throughout:

| Part | Bytes | |
|---|---|---|
| Header | 64 | `PANCELLS`, version (1), statistics, cells per frame (8), reserved; centre of the south-west cell (lat, lon), step (0.1°), rows, columns; first day (days since 1970-01-01), days, scale (100), length of the metadata (`<8sHHHHdddIIiIII`) |
| Offset table | 8 per frame, plus 8 | Where each frame starts, then where the last one ends. The frame of row `r` and column `c` is `r × ceil(columns / 8) + c // 8`: its two offsets lie at byte `64 + 8 × frame` |
| Frames | ~4 KB | 16 bytes: row, first column, cells, days (`<IIII`); then zstd over int32 hundredths (−2³¹ where missing), shaped (statistic, cell, day), byte-stream split: every value's first byte, then every second byte, and so on |
| Metadata | | JSON: variable, unit, statistics, stream, box, licence, attribution |

Reading Warsaw's PM10 of 2021, with numpy and pyarrow:

```python
import struct
import urllib.request

import numpy as np
import pyarrow as pa

URL = (
    "https://huggingface.co/datasets/adrien4916/air_qual_indicators/resolve/main/"
    "cams-europe-v1/cams-pm10-2021-validated.cells"
)


def get(start: int, length: int) -> bytes:
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(request) as answer:
        return answer.read()


header = struct.unpack("<8sHHHHdddIIiIII", get(0, 64))
statistics, per_frame = header[2], header[3]
first_lat, first_lon, step, rows, columns, days = (*header[5:10], header[11])
lat, lon = 52.23, 21.01
row, column = round((lat - first_lat) / step), round((lon - first_lon) / step)
frame = row * -(-columns // per_frame) + column // per_frame
start, end = struct.unpack("<QQ", get(64 + 8 * frame, 16))
data = get(start, end - start)
_, first, cells, _ = struct.unpack("<IIII", data[:16])
raw = pa.Codec("zstd").decompress(data[16:], decompressed_size=statistics * cells * days * 4)
values = np.frombuffer(raw, np.uint8).reshape(4, -1).T.copy().view("<i4")
values = values.reshape(statistics, cells, days)[0, column - first]
pm10 = np.where(values == -(2**31), np.nan, values / 100)  # μg/m³, 1 January first
```

## How it is built

Each variable-year is built from the ADS (the Copernicus Atmosphere Data Store), one month at a time. The steps are the same as Panache's: UTC days, 75 % of the hours, and the 8-hour means carried across months.

Each asset is checked two ways before it is published:
- against Open-Meteo at Lyon, Madrid, Warsaw, Athens and Helsinki (Open-Meteo serves the same reanalysis until 2021);
- against the archive's France assets, whose numbers it must hold exactly.

## Licence and attribution

CC-BY 4.0, as the sources. Contains modified Copernicus Atmosphere Monitoring Service and Copernicus Climate Change Service information [2026]. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains. Source: CAMS European air quality reanalyses, DOI [10.24381/7cc0465a](https://doi.org/10.24381/7cc0465a).
