ERA5's boundary layer over Europe, for [Panache](https://github.com/AdrienJKTH/panache)'s Lagrangian footprints (its D86).

- **Content**: the boundary layer's height (m), the friction velocity (m/s) and the instantaneous surface sensible heat flux (W/m², ECMWF's sign: positive downward), every hour from 00 UTC.
- **Grid**: the `winds-europe-v1` release's: ERA5's 0.25° points one in two each way (0.5°, not averaged), centres from 15° N to 84.5° N and from 70° W to 89.5° E, 140 rows by 320 columns, in the same 10° tiles.
- **Assets**: one per month, `surface-2021-02.surface` (~85 MB), 2013 onwards.
- **Format**: a 128-byte header (with each field's scale and offset), a table of offsets, then one zstd frame per day and 10° tile, of int16 values (the height to the metre, the friction velocity to 0.005 m/s, the heat flux to 2 W/m²). One tile's day is read with two HTTP range requests. `src/panache_archive/surface.py` describes it in full.
- **Source**: ERA5 (Hersbach et al. 2020, doi:10.1002/qj.3803), Copernicus Climate Change Service, read from Google's ARCO-ERA5 (https://github.com/google-research/arco-era5). Contains modified Copernicus Climate Change Service information.
