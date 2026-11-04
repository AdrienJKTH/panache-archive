#!/usr/bin/env bash
# Keeps the major roads of a Geofabrik extract, for `fine-layers.sh`: motorways, trunks, primary
# and secondary roads and their links. Run once for all the blocks, so that Geofabrik serves the
# extract only once.
#
# Usage: scripts/fine-roads.sh EXTRACT OUT.osm.pbf
# EXTRACT is the path on download.geofabrik.de, such as europe-latest.osm.pbf.
set -euo pipefail

extract=$1
out=$2
whole="$out.whole.osm.pbf"

curl -fsSL --retry 4 -o "$whole" "https://download.geofabrik.de/$extract"
ls -l "$whole"
osmium tags-filter -o "$out" "$whole" \
  w/highway=motorway,motorway_link,trunk,trunk_link,primary,primary_link,secondary,secondary_link
rm "$whole"
ls -l "$out"
