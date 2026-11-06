---
title: Panache compute
emoji: 🌫️
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: NO₂ at ~100 m over a map view, computed on request
---

The computation behind Panache's "compute in high definition" button: a day's NO₂ at ~100 m over a view of France, by a LightGBM model that downscales the CAMS European reanalysis with ERA5, CORINE, GHSL, the Copernicus DEM and OpenStreetMap's roads.

`GET /api/map/fine/nitrogen_dioxide/{day}/compute?south&north&west&east` answers with the view as an image and its scale. One view at a time, at most 400,000 cells.

A statistical estimate, not a measurement: on stations it never saw, it reads ~1 µg/m³ high in the background and ~7 µg/m³ low beside roads.

Built and deployed by [AdrienJKTH/panache-archive](https://github.com/AdrienJKTH/panache-archive). Sources: the Copernicus Atmosphere Monitoring Service, the Copernicus Climate Change Service, the EEA's validated measurements, the Copernicus Land Monitoring Service, JRC's GHSL (CC-BY 4.0), the Copernicus DEM GLO-90 (© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA) and OpenStreetMap (© OpenStreetMap contributors, ODbL).
