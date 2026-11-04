#!/usr/bin/env bash
# Keeps Europe's major roads, for `fine-layers.sh`: motorways, trunks, primary and secondary roads
# and their links. Run once for all the blocks, so that Geofabrik serves each extract only once.
# The whole of Europe (30 GB and more) does not fit on a runner's ~14 GB, so its regions are
# fetched one at a time, each kept only as its roads. Regions overlap at borders: merging keeps one
# copy of each road.
#
# Usage: scripts/fine-roads.sh REGION OUT.osm.pbf
# REGION is `europe`, or one of Geofabrik's regions, such as `alps`, to try a block.
set -euo pipefail

region=$1
out=$2
parts="$out.parts"
mkdir -p "$parts"

index=$(curl -fsSL --retry 4 https://download.geofabrik.de/index-v1-nogeom.json)
# Left out: regions that repeat others, and those beyond CORINE or the domain.
urls=$(jq -r --arg region "$region" '.features[].properties
  | select(if $region == "europe"
      then .parent == "europe" and (.id | IN(
        "alps", "britain-and-ireland", "dach", "united-kingdom",
        "ukraine", "belarus", "moldova", "georgia", "azores") | not)
      else .id == $region end)
  | .urls.pbf' <<< "$index")

for url in $urls; do
  name=$(basename "$url" -latest.osm.pbf)
  curl -fsSL --retry 4 -o "$parts/whole.osm.pbf" "$url"
  echo "$name: $(stat -c %s "$parts/whole.osm.pbf") bytes"
  osmium tags-filter -o "$parts/$name.osm.pbf" "$parts/whole.osm.pbf" \
    w/highway=motorway,motorway_link,trunk,trunk_link,primary,primary_link,secondary,secondary_link
  rm "$parts/whole.osm.pbf"
done
osmium merge -o "$out" "$parts"/*.osm.pbf
rm -r "$parts"
ls -l "$out"
df -h "$(dirname "$out")" | tail -1
