#!/usr/bin/env bash
# Downloads CORINE Land Cover 2018 (raster 100 m) over Europe from the Copernicus Land
# Monitoring Service's API, with the service key in $CLMS_SERVICE_KEY: a request cut to the box,
# prepared by the service in the background, polled until it is ready (its link lasts 72 hours).
# https://eea.github.io/clms-api-docs/download.html
#
# Usage: scripts/clms-corine.sh OUT.zip
set -euo pipefail

out=$1
api=https://land.copernicus.eu/api
here=$(dirname "$0")
dataset=0407d497d3c44bcd93ce8fd5bf78596a       # CORINE Land Cover 2018
raster=7bcdf9d1-6ba0-4d4e-afa8-01451c7316cb    # its raster 100 m
west=-25.0 north=72.0 east=45.0 south=30.0     # the domain (src/panache_archive/fine.py)

token=$("$here/clms-token.sh")
echo "::add-mask::$token"
body=$(jq -nc --arg d "$dataset" --arg r "$raster" \
  --argjson box "[$west, $north, $east, $south]" \
  '{Datasets: [{DatasetID: $d, DatasetDownloadInformationID: $r, OutputFormat: "Geotiff",
    OutputGCS: "EPSG:3035", BoundingBox: $box}]}')
task=$(curl -fsS -X POST "$api/@datarequest_post" -H "Authorization: Bearer $token" \
  -H 'Accept: application/json' -H 'Content-Type: application/json' -d "$body" \
  | jq -re '.TaskIds[0].TaskID')
echo "CORINE requested: task $task"

started=$(date +%s)
while true; do
  # A token lasts an hour: a new one each round.
  token=$("$here/clms-token.sh")
  echo "::add-mask::$token"
  status=$(curl -fsS "$api/@datarequest_status_get?TaskID=$task" \
    -H "Authorization: Bearer $token" -H 'Accept: application/json')
  state=$(jq -r .Status <<< "$status")
  echo "$(( ($(date +%s) - started) / 60 )) min: $state"
  case "$state" in
    Finished_ok) break ;;
    Rejected | Cancelled | Finished_nok) echo "$status" >&2; exit 1 ;;
  esac
  sleep 120
done
curl -fsSL --retry 4 -o "$out" "$(jq -r .DownloadURL <<< "$status")"
ls -l "$out"
