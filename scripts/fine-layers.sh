#!/usr/bin/env bash
# Cuts the fixed layers at 3 arc seconds over a block with GDAL, for `panache-archive fine`
# (src/panache_archive/fine.py): GHS-POP and GHS-BUILT-S's 100 m tiles averaged (counts per 100 m
# pixel), the Copernicus DEM GLO-90 averaged, CORINE's class at each cell (nearest), and the
# block's major roads from OpenStreetMap as a GeoJSON text sequence.
#
# Usage: scripts/fine-layers.sh WORKDIR CORINE.zip ROADS.osm.pbf SOUTH NORTH WEST EAST "GHSL TILES"
# Writes WORKDIR/population.tif, built_up.tif, elevation.tif, corine.tif and roads.geojsonseq, or
# WORKDIR/empty alone for a block of sea: no GHSL tile or no DEM tile covers it.
set -euo pipefail

work=$1
corine_zip=$2
roads=$3
south=$4 north=$5 west=$6 east=$7
ghsl_tiles=$8
mkdir -p "$work"

cells=1200  # a degree
cols=$(((east - west) * cells))
rows=$(((north - south) * cells))
grid=(-te "$west" "$south" "$east" "$north" -ts "$cols" "$rows" -co TILED=YES -co COMPRESS=DEFLATE)

# GHSL's 100 m tiles (Mollweide) around the block; those absent from the server are the sea's.
jrc=https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL
ghsl() {  # folder, stem, output name
  local parts=()
  for tile in $ghsl_tiles; do
    local name="$2_$tile"
    if curl -fsSL --retry 4 -o "$work/$name.zip" "$jrc/$1/tiles/$name.zip"; then
      parts+=("/vsizip/$work/$name.zip/$name.tif")
    fi
  done
  if ((${#parts[@]} == 0)); then
    touch "$work/empty"
    exit 0
  fi
  gdalbuildvrt -q "$work/$3.vrt" "${parts[@]}"
  gdalwarp -q -t_srs EPSG:4326 "${grid[@]}" -r average -ot Float32 "$work/$3.vrt" "$work/$3.tif"
  rm -f "$work"/"$2"_R*.zip
}
ghsl GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_54009_100/V1-0 \
  GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0 population
ghsl GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100/V1-0 \
  GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0 built_up

# The DEM's 1° tiles over the block; the sea has none, and gdalbuildvrt skips the missing ones.
tiles=()
for lat in $(seq "$south" $((north - 1))); do
  for lon in $(seq "$west" $((east - 1))); do
    if ((lon < 0)); then side=W degrees=$((-lon)); else side=E degrees=$lon; fi
    name=$(printf "Copernicus_DSM_COG_30_N%02d_00_%s%03d_00_DEM" "$lat" "$side" "$degrees")
    tiles+=("/vsicurl/https://copernicus-dem-90m.s3.amazonaws.com/$name/$name.tif")
  done
done
if ! gdalbuildvrt -q "$work/elevation.vrt" "${tiles[@]}"; then
  touch "$work/empty"
  exit 0
fi
gdalwarp -q "${grid[@]}" -r average -ot Float32 "$work/elevation.vrt" "$work/elevation.tif"

# CORINE's class index (1 to 44) at each cell, from its 100 m raster (EPSG:3035).
mkdir -p "$work/corine"
unzip -q -o "$corine_zip" -d "$work/corine"
clc=$(find "$work/corine" -name '*.tif' | head -1)
gdalwarp -q -t_srs EPSG:4326 "${grid[@]}" -r near -ot Byte "$clc" "$work/corine.tif"
rm -rf "$work/corine"

# The block's major roads, from those `fine-roads.sh` kept.
osmium extract -b "$west,$south,$east,$north" -o "$work/block.osm.pbf" "$roads"
osmium export -f geojsonseq -o "$work/roads.geojsonseq" --geometry-types=linestring \
  "$work/block.osm.pbf"
rm "$work/block.osm.pbf"
ls -l "$work"
