# LAVEO: Lavender Area Vectorisation with Earth Observation

LAVEO is a geospatial AI project for mapping lavender fields from Sentinel-2 satellite imagery using deep learning and geospatial post-processing.

## Workflow

1. **Satellite imagery** — Sentinel-2 imagery acquired during the lavender flowering period.
2. **Deep learning segmentation** — ResUNet predicts lavender areas from satellite image patches.
3. **Raster-to-vector conversion** — Converts binary prediction rasters into polygons.
4. **Area filtering** — Removes polygons smaller than 5,000 m² (0.5 ha).
5. **Geometry simplification** — Simplifies polygon boundaries while preserving topology.

## Study Area

Dobrich Province, Bulgaria

## Technology

- Python
- PyTorch
- Rasterio
- GeoPandas
- Shapely
- Sentinel-2

## Outputs

The project produces georeferenced raster predictions and vector polygon datasets representing detected lavender fields.

## Status

Research and development. The workflow is being evaluated for cross-region transferability and the quality of mapped field boundaries.

