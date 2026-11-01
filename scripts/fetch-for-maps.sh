#!/usr/bin/env bash
# Fetches what Panache's `panache-downscaling-maps` reads into an archive folder laid out as its
# `data/archive`: the CAMS species its features read (release cams-v1), ERA5 (era5-v1) and the
# downscaling's inputs (downscaling-v1).
#
# Usage: scripts/fetch-for-maps.sh DIR YEAR|all
set -euo pipefail

dir=$1
year=$2
pattern=$([[ $year == all ]] && echo "*" || echo "$year")

species=(pm2_5 pm10 nitrogen_dioxide ozone carbon_monoxide secondary_inorganic_aerosol dust)
for name in "${species[@]}"; do
  # A species CAMS lacks that year (dust before 2018) matches no asset: its feature stays missing.
  gh release download cams-v1 --pattern "cams-$name-$pattern-*.parquet" --dir "$dir/cams" \
    --skip-existing || echo "no cams-$name-$pattern"
done
gh release download era5-v1 --pattern "era5-$pattern.parquet" --dir "$dir/era5" --skip-existing
gh release download downscaling-v1 --dir "$dir/downscaling" --skip-existing
du -sh "$dir"/*
