Panache's map of today and the next days over the whole CAMS Europe domain (30-72° N, 25° W-45° E, 420 by 700 cells of 0.1° centred on x.x5°), from CAMS Europe's latest daily run (Panache's D80). These are forecasts, not the reanalysis.

- `live.json`: the latest run, the days it gives and its files. Read it first; a new run replaces it last.
- `map-eaqi-run-<day>.days`: the EAQI band of the day's worst pollutant on each whole UTC day of the run of `<day>`, today to J+3, 1 (Good) to 6 (Extremely poor); 0 unclassed. A run's file never changes once published; runs older than a week are deleted.

Each file: a 64-byte header, an offset table and one zstd frame a day (uint16 classes, bytes split by weight), then JSON with each day's cells per class; the format is described in Panache's `backend/src/panache/downscaling/frames.py`.

Built by the `Live maps` workflow from Open-Meteo's public bucket, by Panache's own rules (`panache-domain-maps`). Generated using Copernicus Atmosphere Monitoring Service information [2026] (CAMS European air quality forecasts), through Open-Meteo; neither the European Commission nor ECMWF is responsible for any use that may be made of the information it contains. CC-BY 4.0.
