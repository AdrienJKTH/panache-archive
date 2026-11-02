#!/usr/bin/env bash
# Cuts the fixed layers at 3 arc seconds over France's box with GDAL, for `panache-archive fine`
# (src/panache_archive/fine.py): GHS-POP and GHS-BUILT-S's 100 m tiles averaged (counts per 100 m
# pixel), the Copernicus DEM GLO-90 averaged, CORINE's class at each cell (nearest), and France's
# major roads from OpenStreetMap as a GeoJSON text sequence.
#
# Usage: scripts/fine-layers.sh WORKDIR CORINE.zip
# Writes WORKDIR/population.tif, built_up.tif, elevation.tif, corine.tif and roads.geojsonseq.
set -euo pipefail

work=$1
corine_zip=$2
mkdir -p "$work"

# France's box (src/panache_archive/grid.py), 1,200 cells a degree.
west=-5.5 south=41.0 east=10.0 north=51.5
cols=18600 rows=12600
grid=(-te "$west" "$south" "$east" "$north" -ts "$cols" "$rows" -co TILED=YES -co COMPRESS=DEFLATE)

# GHSL's 100 m tiles (Mollweide) around France; those absent from the server are skipped.
jrc=https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL
ghsl() {  # folder, stem, output name
  local parts=()
  for row in 3 4 5; do
    for col in 17 18 19 20; do
      local name="$2_R${row}_C${col}"
      if curl -fsSL --retry 4 -o "$work/$name.zip" "$jrc/$1/tiles/$name.zip"; then
        parts+=("/vsizip/$work/$name.zip/$name.tif")
      fi
    done
  done
  gdalbuildvrt -q "$work/$3.vrt" "${parts[@]}"
  gdalwarp -q -t_srs EPSG:4326 "${grid[@]}" -r average -ot Float32 "$work/$3.vrt" "$work/$3.tif"
  rm -f "$work"/"$2"_R*.zip
}
ghsl GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_54009_100/V1-0 \
  GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0 population
ghsl GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100/V1-0 \
  GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0 built_up

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
gdalwarp -q "${grid[@]}" -r average -ot Float32 "$work/elevation.vrt" "$work/elevation.tif"

# CORINE's class index (1 to 44) at each cell, from its 100 m raster (EPSG:3035).
mkdir -p "$work/corine"
unzip -q -o "$corine_zip" -d "$work/corine"
clc=$(find "$work/corine" -name '*.tif' | head -1)
gdalwarp -q -t_srs EPSG:4326 "${grid[@]}" -r near -ot Byte "$clc" "$work/corine.tif"
rm -rf "$work/corine"

# France's major roads: motorways, trunks, primary and secondary roads and their links.
curl -fsSL --retry 4 -o "$work/france.osm.pbf" https://download.geofabrik.de/europe/france-latest.osm.pbf
osmium tags-filter -o "$work/roads.osm.pbf" "$work/france.osm.pbf" \
  w/highway=motorway,motorway_link,trunk,trunk_link,primary,primary_link,secondary,secondary_link
rm "$work/france.osm.pbf"
osmium export -f geojsonseq -o "$work/roads.geojsonseq" --geometry-types=linestring \
  "$work/roads.osm.pbf"
ls -l "$work"
