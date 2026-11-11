ERA5's winds on pressure levels over Europe, for [Panache](https://github.com/AdrienJKTH/panache)'s back-trajectories (its D83).

- **Content**: u and v (m/s) and the pressure velocity omega (Pa/s) on 10 levels (500, 600, 700, 800, 850, 900, 925, 950, 975 and 1000 hPa), and the surface pressure (Pa), every 3 hours from 00 UTC.
- **Grid**: ERA5's 0.25° points one in two each way (0.5°, not averaged), centres from 15° N to 84.5° N and from 70° W to 89.5° E: 140 rows by 320 columns.
- **Assets**: one per month, `winds-2021-02.winds` (~320 MB), 2013 onwards.
- **Format**: a 64-byte header, a table of offsets, then one zstd frame per day and 10° tile, of int16 values (u and v in tenths of m/s, omega in units of 0.002 Pa/s, the surface pressure in units of 2 Pa around 80,000 Pa). One tile's day is read with two HTTP range requests. `src/panache_archive/winds.py` describes it in full.
- **Source**: ERA5 (Hersbach et al. 2020, doi:10.1002/qj.3803), Copernicus Climate Change Service, read from Google's ARCO-ERA5 (https://github.com/google-research/arco-era5). Contains modified Copernicus Climate Change Service information.
