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
- **The two grids** are kept as for France: each file holds its year's own grid, given by its header. Until 2017 the reanalysis stops at 70° N: its files of 2013-2017 have 401 rows, not 421 (Panache's D78).

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

Every release (the whole domain's, France's, and the downscaling's) is mirrored on Hugging Face, in the public dataset [`adrien4916/air_qual_indicators`](https://huggingface.co/datasets/adrien4916/air_qual_indicators) (one folder per release), which Panache reads when GitHub fails (its decisions D66, D71 and D72). After each run that publishes assets and every three hours, the [`Mirror`](.github/workflows/mirror.yml) workflow (`panache-archive mirror`, through `huggingface_hub`):

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
- **Check**: against the archive API at the four places. Temperature agrees once its constant offset is removed. Humidity is judged by correlation (r ≥ 0.99), since the API derives it from altitude-corrected temperature and dew point. Rain is checked against the API's hours summed over the archive's day (01:00 to 24:00), since the API's own daily sum now takes the hours stamped 00:00 to 23:00 (Panache's D62). The other columns agree within rounding.
- The [`ERA5`](.github/workflows/era5.yml) workflow builds the years the release lacks, from 2013 to the last year the bucket holds whole; labelling a pull request `pilot-era5` tries it on January 2021. Where the bucket's hours stop comes from its [`meta.json`](https://openmeteo.s3.amazonaws.com/data/copernicus_era5/static/meta.json) (`data_end_time`, about five days back); a month counts once the bucket holds the next month's first hour, for its last day's rain, and a day more.

**The whole domain** (Panache's D71): the release [`era5-europe-v1`](../../releases/tag/era5-europe-v1) holds the same nine columns over the whole CAMS Europe domain, 30 to 72° N and 25° W to 45° E (169 by 281 points), as `.cells` files such as `era5-2021.cells` ([the format](#the-whole-domain), one point a frame: a point's year is ~6 KB). The domain is read from the bucket in bands of 24 latitudes. Each asset is checked against the archive API at Lyon, Madrid, Warsaw, Athens and Helsinki, and must hold its France twin's numbers exactly.

The whole domain also keeps **the year under way** as its months out so far, such as `era5-2026-01-09.cells` (Panache's D75), so that a click reads this year's ERA5 from the archive too. Each Monday the workflow adds the months the bucket completed since, then deletes the year's earlier part. The last months are ERA5T, ERA5's first release, which ECMWF confirms two to three months later and very rarely corrects. France's archive, downloaded whole, keeps whole years only. Run it by hand with:

```bash
gh workflow run era5.yml -f domain=europe -f years=all
```

## MERRA-2 daily aerosols

The release [`merra2-europe-v1`](../../releases/tag/merra2-europe-v1) holds NASA's MERRA-2 aerosols over the whole CAMS domain (Panache's D72), for the second opinion on an episode's aerosols (its D68): one `.cells` asset per year, such as `merra2-2021.cells`, and the year under way as its months out so far, such as `merra2-2026-01-09.cells`.

| Statistic | Unit | |
|---|---|---|
| `DUSMASS` | μg/m³ | Dust, its five bins up to 20 µm |
| `OCSMASS` | μg/m³ | Organic carbon, from every source |
| `BCSMASS` | μg/m³ | Black carbon, from every source |

- **Source**: MERRA-2 tavg1_2d_aer_Nx (M2T1NXAER) V5.12.4, hourly, 0.5° by 0.625°, doi:[10.5067/KLICLTZ8EM9D](https://doi.org/10.5067/KLICLTZ8EM9D). NASA's Cloud OPeNDAP cuts each daily file to the domain and the three species (~1.8 MB a day), with an Earthdata Login token (the secret `EARTHDATA_TOKEN`). A few months were reprocessed: their files are `MERRA2_401`, read when `MERRA2_400` is not found.
- **Days**: UTC daily means of at least 18 hours.
- **Grid**: 85 by 113 cells, from 30° N every 0.5° and from 25° W every 0.625°: version 2 of the format, for its two steps; one cell a frame.
- **Check**: against Giovanni's hours at Lyon, Madrid, Warsaw, Athens and Helsinki, the same values but for the archive's rounding (0.005 μg/m³) and Giovanni's 6 significant digits (up to 5 millionths of the value, 0.0013 μg/m³ on 259 μg/m³ of dust).
- The [`MERRA-2`](.github/workflows/merra2.yml) workflow builds what the release lacks, and each Monday the months MERRA-2 put out since (it comes out a few weeks after each month), replacing the year under way's asset. Labelling a pull request `pilot-merra2` tries it on February 2021.

```bash
gh workflow run merra2.yml -f years=all
```

## Downscaling inputs

What Panache's downscaling of CAMS to 1 km over France learns from and maps with (Panache's D74), built by the `Downscaling inputs` workflow into the release `downscaling-v1`:

- `stations-fr-<pollutant>.parquet` (`pm2_5`, `nitrogen_dioxide`, `pm10`, `ozone`, `sulphur_dioxide`): one row per sampling point and UTC day (`sampling_point`, `day`, `mean` in µg/m³, `hours`, the day's valid hours), from the EEA's validated hours (E1a), 2013 to the last complete year. `mean` is the statistic the European Air Quality Index classes the pollutant by, named in the metadata (`statistic`) and computed as the CAMS days are (`daily`): the daily mean, from at least 18 valid hours; for ozone, the day's maximum 8-hour mean, from at least 18 valid 8-hour means of at least 6 hours. The files' hours are UTC+1. Each point's station, position, altitude, type and area, from the EEA's metadata, are in the asset's `panache_archive` metadata (`points`).
- `static-fr-1km.npz`: `population` (inhabitants/km², GHS-POP 2020), `built_up` (share of the cell, GHS-BUILT-S 2020) and `elevation` (m, Copernicus DEM GLO-90 averaged; 0 or NaN at sea) and `land` (share of the cell on land, Natural Earth 1:10m land without its lakes) on a 1/120° grid over France's box (1,260 by 1,860 cells, twelve by twelve per CAMS cell, rows from the south), with `lats`, `lons` and `about` (JSON). GDAL cuts the sources onto the grid (`scripts/static-layers.sh`).

The same inputs beyond France (Panache's D78), built by the [`Downscaling inputs, Europe`](.github/workflows/downscaling-europe.yml) workflow into the release `downscaling-europe-v1`, France's staying as it is:

- `stations-<country>-<pollutant>.parquet` for 18 countries (FR, GB, IE, ES, PT, IT, CH, DE, AT, CZ, PL, SE, NO, FI, DK, NL, BE, LU) and four pollutants (not SO₂, which did not beat CAMS over France), every station type kept. The EEA's validated series of Great Britain end in 2019. Besides the hours, a mean takes the value a network reports for the whole day on the days without enough hours (`hours` 0, the file's own day): many networks weigh their particles on a filter, all of Switzerland's and nearly all of Ireland's.
- `static-europe-1km.npz`: the same four layers on a 1/120° grid over 35-71.5° N, 11° W-32° E, 4,380 by 5,160 cells, `land` being the share of each cell on the land of those countries and their small neighbours (Natural Earth's countries, by `ADM0_A3`).

```bash
gh workflow run downscaling-europe.yml
```

Labelling a pull request `pilot-downscaling-europe` builds two countries' PM2.5 and the whole fixed layers as the run's artifacts.

## Domain maps

Panache's map colourings over the whole CAMS Europe domain (Panache's D78), every day from 2013, in the release `maps-europe-v1`: one `map-<layer>-<year>.days` file per colouring and year, ~2 to 20 KB a day, read a day at a time over HTTP ranges (the format of the downscaled maps below, classes stored at a step of 1):

- `eaqi`: the EAQI band of the day's worst pollutant, 1 to 6;
- `unusual`: the most unusual of PM2.5, PM10 and ozone's 8-hour maximum for the season, 1 (up to p5) to 7 (p99 or more);
- `causes`: the leading mechanism of Panache's verdict on the cell's day, 3 to 12; 1 not in an episode, 2 none standing out.

0 means no class. Each file's JSON counts each day's cells per class.

- **Built by Panache's own backend** (`panache-domain-maps`, checked out at the commit the run is given with the deploy key `PANACHE_DEPLOY_KEY`), from this repository's releases alone: nothing comes from the ADS.
- The [`Domain maps`](.github/workflows/maps.yml) workflow shares the 420 rows out among 12 jobs. Each reads its rows of `cams-europe-v1` and `era5-europe-v1` over HTTP ranges (~1 GB), writes them as France's Parquet assets and weighs every day by the rules of Panache's map of France, which it then matches exactly. A last job stacks the bands and publishes them, and the mirror copies them to Hugging Face. With `pilot`, one band is built as the run's artifact and nothing is published.

```bash
gh workflow run maps.yml -f panache=main -f pilot=6 -f years=2021
gh workflow run maps.yml -f panache=main -f years=all
```

## Live maps

Panache's map of today and the next days (Panache's D80), in the release `maps-live-v1`: one `map-eaqi-run-<day>.days` file a run, ~30 KB, holding the EAQI bands of its four whole UTC days, then `live.json`, which names the latest run. These are CAMS's forecasts, not the reanalysis.

- The [`Live maps`](.github/workflows/live.yml) workflow reads CAMS Europe's daily run from Open-Meteo's public bucket, ~50 MB, out about 11:35 UTC, with no account and no quota, and classes it by Panache's own rules (`panache-domain-maps live`).
- It starts at 12:05, 14:05, 17:05 and 21:05 UTC, since GitHub's cron is late at times. A run already published is left alone.
- A run's file never changes once published, and `live.json` is replaced last, so a reader never meets a half-published run. Runs older than a week are deleted.
- GitHub turns scheduled workflows off after 60 days without activity in a repository: Panache shows the run's age, so a stopped workflow can be seen.

```bash
gh workflow run live.yml -f panache=main
```

## Downscaled maps

Daily values of the European Air Quality Index's pollutants at 1 km over the land of France's box (PM2.5 and NO₂ first, then PM10, ozone's daily maximum 8-hour mean and SO₂), 2013 to the last complete year (Panache's D74), in the release `downscaled-v1`: one `downscaled-<pollutant>-<year>.days` file per pollutant and year (~1 MB a day, read a day at a time over HTTP ranges), and each pollutant's model (`downscaling-<pollutant>.lgb.txt`).

- **Built by Panache's own backend** (`panache-downscaling-maps`, checked out at the commit the run is given with the read-only deploy key `PANACHE_DEPLOY_KEY`, Panache's repository being private): each pollutant's model is trained once on the EEA's background stations (release `downscaling-v1`), then each year's days are predicted cell by cell from the archive's CAMS (`cams-v1`), ERA5 (`era5-v1`) and the 1 km fixed layers, ~25 s a day.
- The [`Downscaling experiment`](.github/workflows/experiment.yml) workflow measures a pollutant's model against CAMS alone at stations it never saw, before its maps are built.
- The [`Downscaled maps`](.github/workflows/downscaled.yml) workflow trains the models and builds the pollutant-years the release lacks, among the pollutants asked, one job each; a published model is never replaced. With `days`, a pilot builds each year's first days as the run's artifacts.

```bash
gh workflow run downscaled.yml -f panache=main -f years=2021 -f days=3
gh workflow run downscaled.yml -f panache=main -f years=all
```

## Town maps

Daily maps of a town at 3 arc seconds (~90 by 65 m), 2013 to the last complete year (Panache's D74), in the release `downscaled-towns-v1`: NO₂'s daily mean over Paris first, one `downscaled-<pollutant>-<town>-<year>.days` file per pollutant, town and year, in the format of the downscaled maps, and the model (`downscaling-fine-<pollutant>.lgb.txt`). All of France on this grid could not be stored, so towns are built one at a time. The [`Town maps`](.github/workflows/towns.yml) workflow trains the fine model the release lacks on the EEA's background and traffic stations, then builds the years the release lacks, one job each; with `days`, a pilot builds each year's first days as the run's artifacts. `Mirror` copies them to Hugging Face.

## The compute Space

Panache's button "compute in high definition" calls a free Hugging Face Space, [`adrien4916/panache-compute`](https://huggingface.co/spaces/adrien4916/panache-compute), which computes a view at ~100 m on request (Panache's D74). A public Space's image can be pulled by anyone, and Panache's repository is private, so the [`Compute Space`](.github/workflows/space.yml) workflow (`panache-archive space`) deploys only what `panache.compute_app` and `panache.fetch_archive` import, worked out from their imports, with the Dockerfile and card of [`space/`](space/). The Space builds itself and its data from this repository's public releases.

## Fine layers

Fixed layers at 3 arc seconds (~90 by 65 m) over Europe, as far as CORINE covers it, for Panache's downscaling below 1 km (Panache's D74), in the releases `fine-europe-west-v1` and `fine-europe-east-v1`, split at 15° E since a release holds at most 1,000 assets (`fine-v1` holds France's box, on which the first fine model was trained): one `fine-<south><west>.npz` per 1° tile holding land, rows from the south, ten by ten cells in each 30″ cell of the 1 km layers.

- `corine` (uint8): CORINE Land Cover 2018's class at each cell (the raster's index 1 to 44; `about.corine_codes` gives the CLC codes), requested cut to the box from the Copernicus Land Monitoring Service's API with the secret `CLMS_SERVICE_KEY` (`scripts/clms-corine.sh`). Its objects are at least 25 ha: it tells nothing below ~500 m.
- `population` (uint16, inhabitants/km²) and `built_up` (uint8, 250ths of the cell): GHS-POP and GHS-BUILT-S 2020 at 100 m, averaged onto the grid.
- `elevation` (int16, m): the Copernicus DEM GLO-90.
- `road_major`, `road_primary`, `road_secondary` (uint16, m of road in the cell): OpenStreetMap's motorways and trunks, primary and secondary roads with their links, from Geofabrik's extract of Europe (filtered by osmium, cut into 10 m steps); roads abroad are missing.
- The [`Fine layers`](.github/workflows/fine.yml) workflow fetches CORINE and the roads once, then cuts one block of 10° by 10° a job; labelling a pull request `pilot-fine` builds one block, with the Alps' roads alone, as the run's artifact.

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
