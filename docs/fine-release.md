Fixed layers at 3 arc seconds (~90 by 65 m) over France's box (41-51.5° N, 5.5° W-10° E), for Panache's downscaling below 1 km (Panache's D74): one `.npz` per 1° tile holding land (`fine-N45E004.npz` covers 45-46° N, 4-5° E), rows from the south. Built by the `Fine layers` workflow; described in the [README](https://github.com/AdrienJKTH/panache-archive#fine-layers).

- `corine`: CORINE Land Cover 2018's class at each cell, as the raster's index 1 to 44 (`about.corine_codes` gives the CLC code). © European Union, Copernicus Land Monitoring Service, European Environment Agency (EEA).
- `population` (inhabitants/km²) and `built_up` (250ths of the cell): JRC's GHS-POP and GHS-BUILT-S R2023A, epoch 2020, 100 m (CC-BY 4.0).
- `elevation` (m): Copernicus DEM GLO-90 (© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA).
- `road_major`, `road_primary`, `road_secondary` (m of road in the cell): © OpenStreetMap contributors (ODbL), from Geofabrik's extract of France; roads abroad are missing.
