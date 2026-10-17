# Panache archive

Daily air quality over France since 2013, from the CAMS European air quality reanalyses, for **Panache**, the pollution episode detective.

Panache explains pollution episodes at a place: was a day unusual, and why? It reads each place's history from Open-Meteo, whose free tier allows only a few new places a day. This archive holds the same history for every cell of France at once, so that the history costs no call, and France can be drawn as a map.

## What it holds

One asset per variable and year in the release [`cams-v1`](../../releases/tag/cams-v1), such as `cams-pm10-2021-validated.parquet`:

| Column | Type | |
|---|---|---|
| `lat`, `lon` | float64 | Cell centre, in degrees (cells of 0.1°, centred on x.x5°) |
| `day` | date32 | UTC day |
| `mean` | float32 | Daily mean, μg/m³; null with fewer than 18 hours |
| `max_8h_mean` | float32 | Ozone only: the day's highest running 8-hour mean, μg/m³ |

- **Box**: 41 to 51.5° N, 5.5° W to 10° E (mainland France and Corsica, sea included): 105 by 155 cells, about 10 MB a variable-year (20 for ozone, with its two columns).
- **Variables**, by their Open-Meteo names: `pm10`, `pm2_5`, `nitrogen_dioxide`, `ozone` since 2013; `carbon_monoxide`, `sulphur_dioxide` since 2016; `ammonia`, `dust`, `secondary_inorganic_aerosol` since 2018; `pm10_wildfires` since 2019.
- **Source**: the ensemble median at the surface. The validated reanalysis where it exists (until 2024 today), the interim one after it (`-interim` in the name). An interim year is built again once the validated stream covers it.
- **Days** follow Panache's rules, which are those of EU Directive 2008/50/EC: UTC days, 75 % of the data (18 hours for a daily mean; 6 hours for an 8-hour mean and 18 such means for the day's maximum). Each 8-hour mean belongs to the day it ends; 1 January misses the evening before, as in Panache.
- **Values** are rounded to 0.01 μg/m³: far below the reanalysis's own uncertainty, and the files shrink by half or more.
- Each asset's Parquet metadata (key `panache_archive`) repeats its variable, stream, box, rules, licence and attribution.

## How it is built

The ADS cannot cut its reanalysis to an area: every request returns the whole European grid, 0.8 GB per variable and month, about 1 TB for the archive. So the [`CAMS`](.github/workflows/cams.yml) workflow runs on GitHub's machines:

1. **Plan**: asks the ADS which years each variable offers whole, and lists those the release lacks.
2. **Build**, one job per variable-year: downloads each month, keeps the box, sums up its days, deletes the month, then publishes the asset.
3. **Check**: compares each asset with what Open-Meteo serves at Paris, Lyon, Grenoble and Bordeaux. Both come from the same reanalysis, so their daily values must agree but for rounding: Open-Meteo rounds its hours to 0.1 μg/m³, or to whole μg/m³ for some variables such as ozone, so a day may differ by half that step (plus 0.005 for the archive's own rounding).

Run it from the Actions tab, or:

```bash
gh workflow run cams.yml -f variables=all -f years=all
```

Labelling a pull request `pilot` tries its code on February 2021 (PM10 and ozone) and attaches the assets to the run.

### The supervisor

The [`Supervise`](.github/workflows/supervise.yml) workflow pilots the runs: every 15 minutes, and whenever a CAMS run ends, it

- launches a CAMS run when assets are missing and none is on (a run builds only what the release lacks, so failures are retried);
- leaves out an asset that failed 3 times, until someone looks (the CAMS input `skip`);
- deletes an interim asset once its validated twin is built;
- reports progress in the issue **Archive progress**: every asset in a grid, the pace and the time left, and a comment on each event (a run ended, an asset stuck, a check that disagrees, the archive complete), which GitHub notifies.

Its decisions are `panache-archive supervise`'s, tested like the rest; it keeps its last snapshot in the issue's body.

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
