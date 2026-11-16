#!/usr/bin/env bash
# Natural Earth's admin 0 countries (1:10 million, v5.1.2, public domain) burnt with each
# country's NE_ID on cells ten times finer than the emission inventories' 0.1° cells, over the
# winds archive's box (15-85° N, 70° W-90° E). Writes WORKDIR/countries.tif and
# WORKDIR/countries.geojson for `panache-archive countries`.
set -euo pipefail
work=${1:?usage: countries.sh WORKDIR}
mkdir -p "$work"
ne=https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson
curl -fsSL --retry 4 -o "$work/countries.geojson" "$ne/ne_10m_admin_0_countries.geojson"
gdal_rasterize -q -a NE_ID -init 0 -ot UInt32 -te -70 15 90 85 -ts 16000 7000 \
  -l ne_10m_admin_0_countries "$work/countries.geojson" "$work/countries.tif"
