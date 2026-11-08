Each cell's seasonal edges of Panache's *unusual* colouring over the whole CAMS Europe domain (30-72° N, 25° W-45° E, 420 by 700 cells of 0.1° centred on x.x5°), for every calendar date (Panache's D80). They class the days after the archive, from CAMS's daily stream and latest run, as the archive's days are classed: by where a value falls among the same season's values of every year of the archive.

- `season-<indicator>-<statistic>-p<edge>.days`: for PM2.5 and PM10 (daily mean) and ozone (daily maximum 8-hour mean), and for each edge, 366 frames, one a date of a leap year (2028, 29 February included), µg/m³ at 0.1:
  - `p05` and `p25`: a value up to it is in the lowest 5 % or 25 % of its season;
  - `p75`, `p90`, `p95` and `p99`: a value above it is in the season's top 25 %, 10 %, 5 % or 1 %.
- The reference is every daily value within 15 days of the date in every year of `cams-europe-v1`; with fewer than 90 values, no edge (no value in the frame).

Each file: a 64-byte header, an offset table and one zstd frame a date (uint16 multiples of 0.1, bytes split by weight); the format is described in Panache's `backend/src/panache/downscaling/frames.py`.

Built by the `Seasons` workflow from this repository's `cams-europe-v1`, by Panache's own rules (`panache-domain-maps seasons`). Generated using Copernicus Atmosphere Monitoring Service information (CAMS European air quality reanalyses, doi:10.24381/7cc0465a); neither the European Commission nor ECMWF is responsible for any use that may be made of the information it contains. CC-BY 4.0.
