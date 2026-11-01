Panache's map colourings over the whole CAMS Europe domain (30-72° N, 25° W-45° E, 420 by 700 cells of 0.1° centred on x.x5°), every day from 2013 to the last whole year of the archives (Panache's D78): each cell's EAQI band, how unusual it is for the season, and the mechanism behind it, as Panache's map of France shows them.

- `map-eaqi-<year>.days`: the EAQI band of the day's worst pollutant, 1 (Good) to 6 (Extremely poor); 0 unclassed.
- `map-unusual-<year>.days`: the most unusual of PM2.5, PM10 and ozone's 8-hour maximum against every day within 15 days of the date in the other years, 1 (up to p5) to 7 (p99 or more); 0 without enough reference days.
- `map-causes-<year>.days`: the leading mechanism of Panache's verdict on the cell's day, 3 to 12 (likely, then possible: dust, smoke, local build-up, secondary inorganic aerosol, photochemical ozone); 1 not in an episode, 2 in an episode with no mechanism standing out, 0 no data.

Each file: a 64-byte header, an offset table and one zstd frame a day (uint16 classes, bytes split by weight), then JSON with each day's cells per class; the format is described in Panache's `backend/src/panache/downscaling/frames.py`. Read a day over two HTTP ranges.

Built by the `Domain maps` workflow from this repository's `cams-europe-v1` and `era5-europe-v1`, by Panache's own rules (`panache-domain-maps`). The CAMS reanalysis has no cell north of 70° N before 2018. Generated using Copernicus Atmosphere Monitoring Service information (CAMS European air quality reanalyses, doi:10.24381/7cc0465a) and Copernicus Climate Change Service information (ERA5, through Open-Meteo); neither the European Commission nor ECMWF is responsible for any use that may be made of the information it contains. CC-BY 4.0.
