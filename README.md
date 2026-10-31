# Panache archive

Daily air quality and weather since 2013, from the CAMS European air quality reanalyses and ERA5, for **Panache**, the pollution episode detective: over France, and air quality over the whole CAMS Europe domain.

Panache explains pollution episodes at a place: was a day unusual, and why? It reads each place's history from Open-Meteo, whose free tier allows only a few new places a day, and which serves the reanalysis until 2021 only. This archive holds the reanalysis for every cell at once, so that a history costs no call: France's, which can be drawn as a map, and the whole domain's, read one cell at a time from afar.

## What it holds

One asset per variable and year in the release [`cams-v1`](../../releases/tag/cams-v1), such as `cams-pm10-2021-validated.parquet`:

| Column | Type | |
|---|---|---|
| `lat`, `lon` | float64 | Cell centre, in degrees: cells of 0.1°, centred on x.x0° until 2019 and on x.x5° since 2020 (see below) |
| `day` | date32 | UTC day |
| `mean` | float32 | Daily mean, μg/m³; null with fewer than 18 hours |
| `max_8h_mean` | float32 | Ozone only: the day's highest running 8-hour mean, μg/m³ |

- **Box**: 41 to 51.5° N, 5.5° W to 10° E (mainland France and Corsica, sea included): about 10 MB a variable-year (20 for ozone, with its two columns).
- **Two grids**: the reanalysis moved its grid on 1 January 2020, half a cell north-east. Until 2019 the cells are centred on x.x0° (104 by 154 cells in the box, 41.1 to 51.4° N; Lyon's on 45.8° N, 4.8° E); since 2020 on x.x5° (105 by 155, 41.05 to 51.45° N; Lyon's on 45.75° N, 4.85° E). Each asset keeps its year's own grid. To compare years cell by cell, each of today's cells covers a quarter of four old ones.
- **Variables**, by their Open-Meteo names: `pm10`, `pm2_5`, `nitrogen_dioxide`, `ozone` since 2013; `carbon_monoxide`, `sulphur_dioxide` since 2016; `ammonia`, `dust`, `secondary_inorganic_aerosol` since 2018; `pm10_wildfires` since 2019.
- **Source**: the ensemble median at the surface. The validated reanalysis where it exists (until 2024 today), the interim one after it (`-interim` in the name). An interim year is built again once the validated stream covers it.
- **Days** follow Panache's rules, which are those of EU Directive 2008/50/EC: UTC days, 75 % of the data (18 hours for a daily mean; 6 hours for an 8-hour mean and 18 such means for the day's maximum). Each 8-hour mean belongs to the day it ends; 1 January misses the evening before, as in Panache.
- **Values** are rounded to 0.01 μg/m³: far below the reanalysis's own uncertainty, and the files shrink by half or more.
- Each asset's Parquet metadata (key `panache_archive`) repeats its variable, stream, box, rules, licence and attribution.

## The whole domain

One asset per variable and year in the release [`cams-europe-v1`](../../releases/tag/cams-europe-v1), such as `cams-pm10-2021-validated.cells`: the same daily values as France's assets (the same downloads, rules and rounding; each build checks it holds France's numbers exactly), over the whole CAMS Europe grid, 30 to 72° N and 25° W to 45° E: 420 by 700 cells since 2020, about 150 MB a variable-year.

Such a file is too large to read whole for one place, so it is laid out to give one cell's year in two small requests (Panache's decision D66), little-endian throughout:

| Part | Bytes | |
|---|---|---|
| Header | 64 | `PANCELLS`, version (1), statistics, cells per frame (8), then the grid: centre of the south-west cell, step (0.1°), rows and columns; the first day (days since 1970-01-01), the days, the scale (100) and the length of the metadata |
| Offset table | 8 per frame, plus 8 | Where each frame starts, and where the last one ends. A frame holds 8 cells of a latitude row: the frame of row `r` and column `c` is `r × ceil(columns / 8) + c // 8`, so its two offsets lie at byte `64 + 8 × frame` |
| Frames | ~4 KB each | 16 bytes saying which cells follow (row, first column, cells, days), then zstd over the int32 values in hundredths of μg/m³ (−2³¹ where missing), shaped (statistic, cell, day) and byte-stream split (the first byte of every value, then the second, …) |
| Metadata | | JSON: variable, unit, statistics, stream, box, licence, attribution |

- **Statistics**: the daily mean, and for ozone the daily mean then the highest 8-hour mean, in one file.
- **Reading one cell** (`panache_archive.cells.read_cell`, or from afar with HTTP ranges): read the 16 bytes of its frame's offsets, then the frame; check its 16 first bytes; decompress; un-split; pick the cell.
- **The two grids** are kept as for France: each file holds its year's own grid, given by its header.

## How it is built

The ADS cannot cut its reanalysis to an area: every request returns the whole European grid, 0.8 GB per variable and month, about 1 TB for the archive. So the [`CAMS`](.github/workflows/cams.yml) workflow runs on GitHub's machines:

1. **Plan**: asks the ADS which years each variable offers whole, and lists those the release lacks.
2. **Build**, one job per variable-year: downloads each month, keeps the domain's box, sums up its days band of latitudes by band (a whole month of the grid would take 1.75 GB per array), deletes the month, then publishes the asset.
3. **Check**: compares each asset with what Open-Meteo serves at Paris, Lyon, Grenoble and Bordeaux (Lyon, Madrid, Warsaw, Athens and Helsinki for the whole domain). Both come from the same reanalysis until 2021, so their daily values must agree but for rounding: Open-Meteo rounds its hours to 0.1 μg/m³, or to whole μg/m³ for some variables such as ozone, so a day may differ by half that step (plus 0.005 for the archive's own rounding). A whole-domain asset must also hold its France twin's numbers exactly, every year.

Run it from the Actions tab, or:

```bash
gh workflow run cams.yml -f domain=europe -f variables=all -f years=all
```

Labelling a pull request `pilot` tries its code on February 2021 (PM10 and ozone, over France and the whole domain) and February 2019 (the old grid), with each build's time and peak memory, and attaches the assets to the run.

### The supervisor

The [`Supervise`](.github/workflows/supervise.yml) workflow pilots the runs of each domain: after each build job, whenever a CAMS run ends, and every 15 minutes (GitHub's cron is late at times), it

- launches a CAMS run when assets are missing and none is on (a run builds only what the release lacks, so failures are retried);
- leaves out an asset that failed 3 times, until someone looks (the CAMS input `skip`);
- deletes an interim asset once its validated twin is built;
- reports progress in the domain's issue, **Archive progress** and **Archive progress, whole domain**: every asset in a grid, the pace and the time left, and a comment on each event (a run ended, an asset stuck, a check that disagrees, the archive complete), which GitHub notifies.

Its decisions are `panache-archive supervise`'s, tested like the rest; it keeps its last snapshot in the issue's body.

### The Hugging Face mirror

The whole domain's releases `cams-europe-v1` and `era5-europe-v1` are mirrored on Hugging Face, in the public dataset [`adrien4916/air_qual_indicators`](https://huggingface.co/datasets/adrien4916/air_qual_indicators) (one folder per release), which Panache reads when GitHub fails (its decisions D66 and D71). After each CAMS or ERA5 run and every three hours, the [`Mirror`](.github/workflows/mirror.yml) workflow (`panache-archive mirror`, through `huggingface_hub`):

- uploads every asset the mirror lacks, or holds with another size or SHA-256 (in batches of at most 4 GB, one commit each, each file checked against the release's digest);
- deletes the files the release no longer holds (an interim year replaced by its validated twin);
- keeps the dataset card, [`docs/hugging-face-card.md`](docs/hugging-face-card.md), which describes the format and how to read one cell.

It writes with the repository secret `HF_TOKEN` (a token of the dataset's owner, with write access).

## ERA5 daily weather

The release [`era5-v1`](../../releases/tag/era5-v1) holds one asset per year, such as `era5-2021.parquet`: France's daily weather from ERA5 (0.25°, 43 by 63 grid points, the same box), with the nine daily variables of Panache's weather model:

| Column | Unit | |
|---|---|---|
| `temperature_2m_mean` | °C | The grid's: Open-Meteo's archive API adds an altitude correction, a constant at a place (+1.22 °C at Lyon) |
| `relative_humidity_2m_mean` | % | From hourly temperature and dew point (Magnus) |
| `wind_speed_10m_mean` | m/s | From each hour's components |
| `wind_direction_10m_dominant` | ° | Of the day's mean wind, where it blows from |
| `pressure_msl_mean` | hPa | |
| `precipitation_sum` | mm | The 24 hours ending in the day (01:00 to 24:00 UTC): ERA5's accumulations end at their time stamp |
| `shortwave_radiation_sum` | MJ/m² | Same hours |
| `cloud_cover_mean` | % | |
| `boundary_layer_height` | m | The day's mean, as Panache averages it |

- **Source**: Open-Meteo keeps ERA5 hourly in its public bucket on AWS Open Data, read over HTTPS with no account and no quota (`year_YYYY.om` files, then 504-hour chunks), with [omfiles](https://github.com/open-meteo/python-omfiles).
- **Days**: UTC; a mean needs 18 hours, a total all 24, as in the CAMS assets.
- **Check**: against the archive API at the four places. Temperature agrees once its constant offset is removed. Humidity is judged by correlation (r ≥ 0.99), since the API derives it from altitude-corrected temperature and dew point. The other columns agree within rounding.
- The [`ERA5`](.github/workflows/era5.yml) workflow builds the years the release lacks, from 2013 to the last complete year; labelling a pull request `pilot-era5` tries it on January 2021.

**The whole domain** (Panache's D71): the release [`era5-europe-v1`](../../releases/tag/era5-europe-v1) holds the same nine columns over the whole CAMS Europe domain, 30 to 72° N and 25° W to 45° E (169 by 281 points), as `.cells` files such as `era5-2021.cells` ([the format](#the-whole-domain), one point a frame: a point's year is ~6 KB). The domain is read from the bucket in bands of 24 latitudes. Each asset is checked against the archive API at Lyon, Madrid, Warsaw, Athens and Helsinki, and must hold its France twin's numbers exactly. Run it with:

```bash
gh workflow run era5.yml -f domain=europe -f years=all
```

## Development

Python 3.12 with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run ruff check && uv run ruff format --check && uv run mypy && uv run pytest --cov
```

A build on your own machine needs an ADS key (`~/.cdsapirc`, or `CDSAPI_URL` and `CDSAPI_KEY`) and downloads 0.8 GB a month:

```bash
uv run panache-archive build --variable pm10 --year 2021 --months 2
```

## Licence and attribution

The code is under the MIT licence. The data are derived from the CAMS European air quality reanalyses ([DOI 10.24381/7cc0465a](https://doi.org/10.24381/7cc0465a)), under the CC-BY 4.0 licence, and are shared under the same licence:

> Contains modified Copernicus Atmosphere Monitoring Service information [2026]. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

ERA5 assets are derived from ERA5 hourly data on single levels ([DOI 10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47)), CC-BY 4.0, as stored by [Open-Meteo](https://open-meteo.com/) (CC-BY 4.0):

> Contains modified Copernicus Climate Change Service information [2026]. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains. Weather data by Open-Meteo.com.
