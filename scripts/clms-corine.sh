#!/usr/bin/env bash
# Downloads CORINE Land Cover 2018 (raster 100 m) over Europe from the Copernicus Land
# Monitoring Service's API, with the service key in $CLMS_SERVICE_KEY: its pan-European file (a
# box this large is refused), prepared by the service in the background, polled until it is ready
# (its link lasts 72 hours).
# https://eea.github.io/clms-api-docs/download.html
#
# Usage: scripts/clms-corine.sh OUT.zip
set -euo pipefail

out=$1
api=https://land.copernicus.eu/api
here=$(dirname "$0")
dataset=0407d497d3c44bcd93ce8fd5bf78596a       # CORINE Land Cover 2018
raster=3d9e2413-46ca-4f6e-abf8-b6eb429a4e48    # its raster 100 m, ~125 MB

token=$("$here/clms-token.sh")
echo "::add-mask::$token"
body=$(jq -nc --arg d "$dataset" --arg f "$raster" '{Datasets: [{DatasetID: $d, FileID: $f}]}')
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
