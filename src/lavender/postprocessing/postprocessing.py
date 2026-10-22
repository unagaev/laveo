"""
Initial lavender raster-to-vector post-processing pipeline.

Run from the project root:
    python -m lavender.postprocessing.pipeline

Expected input filenames contain each year (2021, 2023, 2025).
By default, inputs are read from data/output/predictions; change INPUT_DIR
if your prediction rasters are stored elsewhere.

The script preserves each source raster CRS for GeoPackage outputs.
Area/width/shape operations are performed in an estimated local projected CRS.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union


# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path.cwd()
INPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "postprocessing"

YEARS = (2021, 2023, 2025)

@dataclass(frozen=True)
class Settings:
    # Expected nominal pixel size. The actual raster transform is authoritative.
    expected_resolution_m: float = 10.0
    min_area_m2: float = 5_000.0
    # Global-width proxy from minimum rotated rectangle. Set to 0 to disable.
    min_estimated_width_m: float = 10.0
    # Simplification tolerance; set to 0 to disable.
    simplify_tolerance_m: float = 2.0
    # Selective concavity repair: only fill small hull-gap components meeting
    # ALL these limits. Set max_concavity_area_m2=0 to disable.
    max_concavity_area_m2: float = 200.0
    max_concavity_width_m: float = 10.0
    max_concavity_area_fraction: float = 0.02
    # Four-connectivity avoids merging diagonally touching pixels.
    connectivity: int = 4
    # Binary prediction values expected: 0=background, 1=lavender.
    lavender_value: int = 1
    # GeoPackage is always written; GeoJSON is optional and disabled by default.
    write_geojson: bool = False

SETTINGS = Settings()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger("lavender.postprocessing")


def _safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def find_year(path: Path) -> int | None:
    for year in YEARS:
        if str(year) in path.name:
            return year
    return None


def projected_crs_for(gdf: gpd.GeoDataFrame):
    """Choose a local projected CRS for metric operations."""
    if gdf.crs is None:
        raise ValueError("Vector data has no CRS.")
    if gdf.crs.is_projected:
        # Use the source projected CRS when its units are metres.
        try:
            axis_units = [axis.unit_name.lower() for axis in gdf.crs.axis_info]
            if axis_units and all("metre" in unit or "meter" in unit for unit in axis_units):
                return gdf.crs
        except Exception:
            pass
    return gdf.to_crs("EPSG:4326").estimate_utm_crs()


def polygonise_raster(raster_path: Path) -> tuple[gpd.GeoDataFrame, Any, dict]:
    """Polygonise only lavender pixels, preserving source CRS and georeferencing."""
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"{raster_path.name} has no CRS; cannot preserve CRS.")
        band = src.read(1, masked=False)
        valid = np.ones(band.shape, dtype=bool)
        if src.nodata is not None:
            valid &= band != src.nodata
        if np.issubdtype(band.dtype, np.floating):
            valid &= np.isfinite(band)

        pixel_size_x = abs(float(src.transform.a))
        pixel_size_y = abs(float(src.transform.e))
        if not (
            np.isclose(pixel_size_x, SETTINGS.expected_resolution_m, atol=0.05)
            and np.isclose(pixel_size_y, SETTINGS.expected_resolution_m, atol=0.05)
        ):
            LOG.warning(
                "%s: nominal pixel size is %.3f x %.3f map units; expected %.1f m. "
                "Check CRS units and raster transform before interpreting area/width thresholds.",
                raster_path.name, pixel_size_x, pixel_size_y, SETTINGS.expected_resolution_m,
            )

        values = np.unique(band[valid])
        allowed = {0, SETTINGS.lavender_value}
        unexpected = set(values.tolist()) - allowed
        if unexpected:
            raise ValueError(
                f"{raster_path.name} is not a binary mask. Found values "
                f"{sorted(unexpected)[:20]}; expected 0 and {SETTINGS.lavender_value}."
            )

        lavender = valid & (band == SETTINGS.lavender_value)
        if not lavender.any():
            empty = gpd.GeoDataFrame(
                {"field_id": [], "geometry": []}, geometry="geometry", crs=src.crs
            )
            metadata = {
                "raster_width": src.width,
                "raster_height": src.height,
                "crs": src.crs.to_string(),
                "pixel_size_x": pixel_size_x,
                "pixel_size_y": pixel_size_y,
                "lavender_pixels": 0,
            }
            return empty, src.crs, metadata

        # Cast only valid mask pixels to 0/1 to avoid NaN-to-integer warnings.
        binary = np.where(lavender, SETTINGS.lavender_value, 0).astype(np.uint8)
        geoms = []
        for geom_mapping, value in shapes(
            binary,
            mask=lavender,
            transform=src.transform,
            connectivity=SETTINGS.connectivity,
        ):
            if int(value) == SETTINGS.lavender_value:
                geom = shape(geom_mapping)
                if not geom.is_empty and geom.area > 0:
                    geoms.append(geom)

        gdf = gpd.GeoDataFrame(
            {"field_id": np.arange(1, len(geoms) + 1, dtype=int)},
            geometry=geoms,
            crs=src.crs,
        )
        metadata = {
            "raster_width": src.width,
            "raster_height": src.height,
            "crs": src.crs.to_string(),
            "pixel_size_x": abs(src.transform.a),
            "pixel_size_y": abs(src.transform.e),
            "lavender_pixels": int(lavender.sum()),
        }
    return gdf, gdf.crs, metadata


def geometry_metrics(gdf: gpd.GeoDataFrame, metric_crs) -> gpd.GeoDataFrame:
    """Add metric area, compactness and a global minimum-rectangle width proxy."""
    result = gdf.copy()
    if result.empty:
        result["area_m2"] = pd.Series(dtype=float)
        result["estimated_width_m"] = pd.Series(dtype=float)
        result["compactness"] = pd.Series(dtype=float)
        return result

    metric = result.to_crs(metric_crs)
    areas, widths, compactnesses = [], [], []
    for geom in metric.geometry:
        if geom is None or geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
            areas.append(0.0)
            widths.append(0.0)
            compactnesses.append(0.0)
            continue
        area = float(geom.area)
        perimeter = float(geom.length)
        # The width of the shortest side of the minimum rotated rectangle is a
        # global proxy. It is not a local minimum-width measurement.
        rect = geom.minimum_rotated_rectangle
        if rect.is_empty or rect.geom_type != "Polygon":
            width = 0.0
        else:
            coords = list(rect.exterior.coords)
            edge_lengths = [
                float(np.hypot(coords[i + 1][0] - coords[i][0],
                               coords[i + 1][1] - coords[i][1]))
                for i in range(len(coords) - 1)
            ]
            positive = [length for length in edge_lengths if length > 0]
            width = min(positive) if positive else 0.0
        compactness = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
        areas.append(area)
        widths.append(width)
        compactnesses.append(compactness)

    result["area_m2"] = areas
    result["estimated_width_m"] = widths
    result["compactness"] = compactnesses
    return result


def stage_summary(year: int, stage: str, gdf: gpd.GeoDataFrame) -> dict:
    areas = gdf["area_m2"] if "area_m2" in gdf.columns else pd.Series(dtype=float)
    return {
        "year": year,
        "stage": stage,
        "feature_count": int(len(gdf)),
        "total_area_ha": float(areas.sum() / 10_000) if len(areas) else 0.0,
        "median_area_ha": float(areas.median() / 10_000) if len(areas) else 0.0,
        "invalid_geometries": int((~gdf.geometry.is_valid).sum()) if len(gdf) else 0,
        "empty_geometries": int(gdf.geometry.is_empty.sum()) if len(gdf) else 0,
    }


def write_gpkg(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    # GeoPackage is the canonical output format.
    gdf.to_file(path, driver="GPKG", index=False)


def _polygon_parts(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        parts = []
        for item in geom.geoms:
            parts.extend(_polygon_parts(item))
        return parts
    return []


def selective_concavity_repair(
    geom,
    *,
    max_area_m2: float,
    max_width_m: float,
    max_area_fraction: float,
):
    """
    Fill only small components of (convex hull - polygon).

    This deliberately conservative heuristic leaves larger concavities intact.
    It does not apply a convex hull to the whole field. The area, width and
    relative-area limits are configurable and should be checked visually.
    """
    if max_area_m2 <= 0 or geom is None or geom.is_empty:
        return geom
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        return geom

    original_area = float(geom.area)
    if original_area <= 0:
        return geom

    gaps = geom.convex_hull.difference(geom)
    candidates = _polygon_parts(gaps)
    if not candidates:
        return geom

    selected = []
    for gap in candidates:
        if gap.area <= 0 or gap.area > max_area_m2:
            continue
        if gap.area / original_area > max_area_fraction:
            continue
        rect = gap.minimum_rotated_rectangle
        if rect.is_empty or rect.geom_type != "Polygon":
            continue
        coords = list(rect.exterior.coords)
        edge_lengths = [
            float(np.hypot(coords[i + 1][0] - coords[i][0],
                           coords[i + 1][1] - coords[i][1]))
            for i in range(len(coords) - 1)
        ]
        positive = [length for length in edge_lengths if length > 0]
        gap_width = min(positive) if positive else 0.0
        if max_width_m > 0 and gap_width > max_width_m:
            continue
        selected.append(gap)

    if not selected:
        return geom

    repaired = unary_union([geom, *selected])
    # Guard against unexpected geometry changes; make valid only when needed.
    if repaired.is_empty or repaired.geom_type not in ("Polygon", "MultiPolygon"):
        return geom
    if not repaired.is_valid:
        repaired = repaired.buffer(0)
    return repaired if not repaired.is_empty else geom


def process_raster(raster_path: Path, year: int, report_rows: list[dict]) -> dict:
    out_dir = OUTPUT_ROOT / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, source_crs, raster_meta = polygonise_raster(raster_path)
    if raw.empty:
        LOG.warning("%s: no lavender pixels found; writing empty GeoPackage outputs.", raster_path.name)
        raw["area_m2"] = pd.Series(dtype=float)
        raw["estimated_width_m"] = pd.Series(dtype=float)
        raw["compactness"] = pd.Series(dtype=float)
        write_gpkg(raw, out_dir / "01_raw_polygons.gpkg")
        write_gpkg(raw, out_dir / "02_area_filtered.gpkg")
        write_gpkg(raw, out_dir / "03_width_filtered.gpkg")
        write_gpkg(raw, out_dir / "04_concavity_repaired.gpkg")
        write_gpkg(raw, out_dir / "05_simplified_final_fields.gpkg")
        for stage in ("raw", "area_filtered", "width_filtered", "concavity_repaired", "final"):
            report_rows.append(stage_summary(year, stage, raw))
        return {"year": year, "input": str(raster_path), "source_crs": str(source_crs),
                "raster": raster_meta, "empty_prediction": True}

    metric_crs = projected_crs_for(raw)
    LOG.info("%s: source CRS=%s; metric CRS=%s", year, source_crs, metric_crs)

    # Stage 1: raw vectorisation.
    raw = geometry_metrics(raw, metric_crs)
    write_gpkg(raw, out_dir / "01_raw_polygons.gpkg")
    report_rows.append(stage_summary(year, "raw", raw))

    # Stage 2: remove invalid/empty geometries and small polygons.
    area_filtered = raw[
        raw.geometry.notna()
        & ~raw.geometry.is_empty
        & raw.geometry.is_valid
        & (raw["area_m2"] >= SETTINGS.min_area_m2)
    ].copy()
    write_gpkg(area_filtered, out_dir / "02_area_filtered.gpkg")
    report_rows.append(stage_summary(year, "area_filtered", area_filtered))

    # Stage 3: filter thin objects using the global width proxy.
    if SETTINGS.min_estimated_width_m > 0:
        width_filtered = area_filtered[
            area_filtered["estimated_width_m"] >= SETTINGS.min_estimated_width_m
        ].copy()
    else:
        width_filtered = area_filtered.copy()
    write_gpkg(width_filtered, out_dir / "03_width_filtered.gpkg")
    report_rows.append(stage_summary(year, "width_filtered", width_filtered))

    # Stage 4: selective concavity repair in a metric CRS.
    repaired = width_filtered.to_crs(metric_crs).copy()
    repaired["geometry"] = repaired.geometry.apply(
        lambda geom: selective_concavity_repair(
            geom,
            max_area_m2=SETTINGS.max_concavity_area_m2,
            max_width_m=SETTINGS.max_concavity_width_m,
            max_area_fraction=SETTINGS.max_concavity_area_fraction,
        )
    )
    repaired = repaired.set_geometry("geometry")
    repaired = repaired.to_crs(source_crs)
    repaired = geometry_metrics(repaired, metric_crs)
    repaired = repaired[
        repaired.geometry.notna() & ~repaired.geometry.is_empty & repaired.geometry.is_valid
    ].copy()
    write_gpkg(repaired, out_dir / "04_concavity_repaired.gpkg")
    report_rows.append(stage_summary(year, "concavity_repaired", repaired))

    # Stage 5: topology-preserving simplification, in metres.
    final = repaired.to_crs(metric_crs).copy()
    if SETTINGS.simplify_tolerance_m > 0:
        final["geometry"] = final.geometry.simplify(
            SETTINGS.simplify_tolerance_m, preserve_topology=True
        )
    final = final.to_crs(source_crs)
    final = geometry_metrics(final, metric_crs)
    final = final[
        final.geometry.notna() & ~final.geometry.is_empty & final.geometry.is_valid
    ].copy()
    write_gpkg(final, out_dir / "05_simplified_final_fields.gpkg")
    # Convenient stable alias for downstream use.
    write_gpkg(final, out_dir / "final_fields.gpkg")
    if SETTINGS.write_geojson:
        final.to_crs("EPSG:4326").to_file(out_dir / "final_fields.geojson", driver="GeoJSON")
    report_rows.append(stage_summary(year, "final", final))

    return {
        "year": year,
        "input": str(raster_path),
        "source_crs": str(source_crs),
        "metric_crs": str(metric_crs),
        "raster": raster_meta,
        "settings": asdict(SETTINGS),
        "counts": {
            "raw": len(raw),
            "area_filtered": len(area_filtered),
            "width_filtered": len(width_filtered),
            "concavity_repaired": len(repaired),
            "final": len(final),
        },
    }


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {INPUT_DIR}\n"
            "Set INPUT_DIR at the top of this script to the folder containing "
            "the 2021, 2023 and 2025 binary prediction rasters."
        )

    rasters = [
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"} and find_year(p) is not None
    ]
    by_year: dict[int, Path] = {}
    for path in sorted(rasters):
        year = find_year(path)
        if year is not None:
            if year in by_year:
                raise ValueError(
                    f"More than one raster found for {year}: {by_year[year].name}, {path.name}. "
                    "Keep one prediction raster per year or adjust the selection logic."
                )
            by_year[year] = path

    missing = sorted(set(YEARS) - set(by_year))
    if missing:
        raise FileNotFoundError(
            f"Missing prediction raster(s) for year(s) {missing} in {INPUT_DIR}. "
            "Check INPUT_DIR and make sure filenames contain the year."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict] = []
    runs = []
    for year in YEARS:
        LOG.info("Processing %s", year)
        runs.append(process_raster(by_year[year], year, report_rows))

    report_path = OUTPUT_ROOT / "validation_summary.csv"
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    metadata = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_ROOT),
        "years_processed": list(YEARS),
        "settings": asdict(SETTINGS),
        "notes": [
            "Raster resolution is checked/reported; raster geotransform is authoritative.",
            "Minimum width is a global minimum-rotated-rectangle proxy, not local minimum width.",
            "Concavity repair fills only small components of convex-hull gaps meeting configured limits.",
            "This is geometric QA, not an accuracy assessment against reference labels.",
            "GeoPackage geometries are saved in each input raster's CRS.",
        ],
        "runs": runs,
    }
    with (OUTPUT_ROOT / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    LOG.info("Finished. Summary: %s", report_path)
    LOG.info("Outputs: %s", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
