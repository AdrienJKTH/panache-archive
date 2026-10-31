#!/usr/bin/env bash
# Cuts and averages the downscaling's fixed layers onto France's 30" grid with GDAL, for
# `panache-archive static` (src/panache_archive/static.py): GHS-POP and GHS-BUILT-S from JRC's
# zips (~480 and ~190 MB, deleted once cut), the Copernicus DEM GLO-90 read from its AWS bucket.
#
# Usage: scripts/static-layers.sh WORKDIR
# Writes WORKDIR/population.tif, built_up.tif and elevation.tif, 1,860 by 1,260 pixels.
set -euo pipefail

work=$1
mkdir -p "$work"

# France's box (src/panache_archive/grid.py), 120 cells a degree.
west=-5.5 south=41.0 east=10.0 north=51.5
cols=1860 rows=1260

# GHSL's 30" grid lies ~0.05 pixel off whole multiples of 1/120°: its counts (residents, built-up
# m²) are summed onto the box's own grid, which keeps their totals.
jrc=https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL
cut_ghsl() {  # folder, file stem, output name
  curl -fsSL --retry 4 -o "$work/$2.zip" "$jrc/$1/$2.zip"
  gdalwarp -q -te "$west" "$south" "$east" "$north" -ts "$cols" "$rows" -r sum -ot Float32 \
    -co COMPRESS=NONE "/vsizip/$work/$2.zip/$2.tif" "$work/$3.tif"
  rm "$work/$2.zip"
}
cut_ghsl GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_4326_30ss/V1-0 \
  GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0 population
cut_ghsl GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss/V1-0 \
  GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0 built_up

# The DEM's 1° tiles over the box; the sea has none, and gdalbuildvrt skips the missing ones.
tiles=()
for lat in $(seq 41 51); do
  for lon in $(seq -6 9); do
    if ((lon < 0)); then side=W degrees=$((-lon)); else side=E degrees=$lon; fi
    name=$(printf "Copernicus_DSM_COG_30_N%02d_00_%s%03d_00_DEM" "$lat" "$side" "$degrees")
    tiles+=("/vsicurl/https://copernicus-dem-90m.s3.amazonaws.com/$name/$name.tif")
  done
done
gdalbuildvrt -q "$work/elevation.vrt" "${tiles[@]}"
gdalwarp -q -te "$west" "$south" "$east" "$north" -ts "$cols" "$rows" -r average -ot Float32 \
  -dstnodata nan -co COMPRESS=NONE "$work/elevation.vrt" "$work/elevation.tif"
ls -l "$work"/*.tif
