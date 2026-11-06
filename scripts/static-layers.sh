#!/usr/bin/env bash
# Cuts and averages the downscaling's fixed layers onto a region's 30" grid with GDAL, for
# `panache-archive static` (src/panache_archive/static.py): GHS-POP and GHS-BUILT-S from JRC's
# zips (~480 and ~190 MB, deleted once cut), the Copernicus DEM GLO-90 read from its AWS bucket,
# and Natural Earth's land and lakes (1:10 million) for the share of each cell on land.
#
# Usage: scripts/static-layers.sh WORKDIR [france|europe]
# Writes WORKDIR/population.tif, built_up.tif, elevation.tif and land.tif, on the region's grid.
set -euo pipefail

work=$1
region=${2:-france}
mkdir -p "$work"

# The regions' boxes (src/panache_archive/grid.py), 120 cells a degree, and the DEM's 1° tiles
# over them: their south-west corners, the last ones included.
case "$region" in
  france)
    west=-5.5 south=41.0 east=10.0 north=51.5
    cols=1860 rows=1260
    tile_lats=$(seq 41 51) tile_lons=$(seq -6 9)
    ;;
  europe)
    west=-11.0 south=35.0 east=32.0 north=71.5
    cols=5160 rows=4380
    tile_lats=$(seq 35 71) tile_lons=$(seq -11 31)
    ;;
  *)
    echo "no region $region" >&2
    exit 2
    ;;
esac

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
# GLO-90's pixels widen north of 50°: the mosaic keeps the finest, and the average does the rest.
tiles=()
for lat in $tile_lats; do
  for lon in $tile_lons; do
    if ((lon < 0)); then side=W degrees=$((-lon)); else side=E degrees=$lon; fi
    name=$(printf "Copernicus_DSM_COG_30_N%02d_00_%s%03d_00_DEM" "$lat" "$side" "$degrees")
    tiles+=("/vsicurl/https://copernicus-dem-90m.s3.amazonaws.com/$name/$name.tif")
  done
done
printf '%s\n' "${tiles[@]}" > "$work/tiles.txt"
gdalbuildvrt -q -resolution highest -input_file_list "$work/tiles.txt" "$work/elevation.vrt"
gdalwarp -q -te "$west" "$south" "$east" "$north" -ts "$cols" "$rows" -r average -ot Float32 \
  -dstnodata nan -co COMPRESS=NONE "$work/elevation.vrt" "$work/elevation.tif"

# Land: Natural Earth's polygons burnt on cells ten times finer, lakes burnt back out, averaged.
ne=https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson
curl -fsSL --retry 4 -o "$work/lakes.geojson" "$ne/ne_10m_lakes.geojson"
if [[ $region == europe ]]; then
  # Only the countries mapped, by ADM0_A3 (ISO_A2 is -99 for France and Norway): those Adrien
  # chose, and the small territories that would be holes among them.
  countries="'FRA','GBR','IRL','ESP','PRT','ITA','CHE','DEU','AUT','CZE','POL','SWE','NOR','FIN'"
  countries+=",'DNK','NLD','BEL','LUX','AND','LIE','MCO','SMR','VAT','IMN','JEY','GGY','GIB'"
  countries+=",'ALD','FRO'"
  curl -fsSL --retry 4 -o "$work/land.geojson" "$ne/ne_10m_admin_0_countries.geojson"
  where=(-l ne_10m_admin_0_countries -where "ADM0_A3 IN ($countries)")
else
  curl -fsSL --retry 4 -o "$work/land.geojson" "$ne/ne_10m_land.geojson"
  where=(-l ne_10m_land)
fi
gdal_rasterize -q -burn 1 -init 0 -ot Byte -te "$west" "$south" "$east" "$north" \
  -ts $((cols * 10)) $((rows * 10)) "${where[@]}" "$work/land.geojson" "$work/land-fine.tif"
gdal_rasterize -q -burn 0 -l ne_10m_lakes "$work/lakes.geojson" "$work/land-fine.tif"
gdalwarp -q -te "$west" "$south" "$east" "$north" -ts "$cols" "$rows" -r average -ot Float32 \
  -co COMPRESS=NONE "$work/land-fine.tif" "$work/land.tif"
rm "$work/land-fine.tif" "$work"/*.geojson
ls -l "$work"/*.tif
