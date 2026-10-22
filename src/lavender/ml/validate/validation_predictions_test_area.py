"""
Evaluate France lavender predictions against manually mapped fields.

For each year:
    - Prediction: model output GeoTIFF
    - Reference: manually mapped lavender fields in GeoPackage

The reference polygons are rasterized onto the exact grid of the
prediction raster, then pixel-level metrics are calculated.

Metrics:
    - IoU
    - Dice / F1
    - Precision
    - Recall
    - Accuracy
    - TP / FP / FN / TN
    - Reference area (ha)
    - Predicted area (ha)
    - Overlap area (ha)

IMPORTANT:
    The prediction raster and reference GeoPackage must represent
    the same geographic test area/year.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
import geopandas as gpd


# ============================================================
# CONFIG
# ============================================================

BASE = Path(
    r"C:\Users\rost8\OneDrive\Desktop\Earth_Observation_Intelligence_Platform\data"
)

PREDICTION_DIR = BASE / "tiles" / "validation_france"
LABEL_DIR = BASE / "labels"

# Same threshold used in your Bulgarian independent evaluation
THRESHOLD = 0.5


# ------------------------------------------------------------
# IMPORTANT:
# Replace the 2021 prediction filename with the actual file.
# ------------------------------------------------------------

YEARS = {
    2021: {
        "prediction": PREDICTION_DIR / "val_2020-07-02.tif",
        "labels": LABEL_DIR / "Fr_Lavender fields-07-2021.gpkg",
    },

    2023: {
        "prediction": PREDICTION_DIR / "val_2023-07-02.tif",
        "labels": LABEL_DIR / "Fr_Lavender fields-07-2023.gpkg",
    },

    2025: {
        "prediction": PREDICTION_DIR / "val_2025-06-26.tif",
        "labels": LABEL_DIR / "Fr_Lavender fields-07-2025.gpkg",
    },
}


OUTPUT_CSV = BASE / "france_validation_results.csv"


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(reference, prediction):
    """
    Calculate pixel-level binary segmentation metrics.
    """

    reference = reference.astype(bool)
    prediction = prediction.astype(bool)

    tp = np.logical_and(reference, prediction).sum()
    fp = np.logical_and(~reference, prediction).sum()
    fn = np.logical_and(reference, ~prediction).sum()
    tn = np.logical_and(~reference, ~prediction).sum()

    # --------------------------------------------------------
    # Dice / F1
    # --------------------------------------------------------

    dice_denominator = 2 * tp + fp + fn

    if dice_denominator > 0:
        dice = 2 * tp / dice_denominator
    else:
        dice = 0.0

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou_denominator = tp + fp + fn

    if iou_denominator > 0:
        iou = tp / iou_denominator
    else:
        iou = 0.0

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision_denominator = tp + fp

    if precision_denominator > 0:
        precision = tp / precision_denominator
    else:
        precision = 0.0

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall_denominator = tp + fn

    if recall_denominator > 0:
        recall = tp / recall_denominator
    else:
        recall = 0.0

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    total = tp + fp + fn + tn

    if total > 0:
        accuracy = (tp + tn) / total
    else:
        accuracy = 0.0

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
        "Accuracy": float(accuracy),
    }


# ============================================================
# LOAD REFERENCE
# ============================================================

def load_reference_labels(label_path, target_crs):
    """
    Load manually mapped lavender fields from a GeoPackage.

    Reprojects the reference data to the prediction raster CRS.
    """

    label_path = Path(label_path)

    if not label_path.exists():
        raise FileNotFoundError(
            f"Reference file not found:\n{label_path}"
        )

    print(f"Loading reference: {label_path}")

    # Read GeoPackage
    gdf = gpd.read_file(label_path)

    if gdf.empty:
        raise ValueError(
            f"Reference file contains no features:\n{label_path}"
        )

    if gdf.crs is None:
        raise ValueError(
            f"Reference file has no CRS:\n{label_path}"
        )

    print(f"Reference CRS:    {gdf.crs}")
    print(f"Reference fields: {len(gdf)}")

    # Reproject to prediction raster CRS
    if gdf.crs != target_crs:

        print(
            f"Reprojecting reference:\n"
            f"    {gdf.crs} -> {target_crs}"
        )

        gdf = gdf.to_crs(target_crs)

    # Remove empty geometries
    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    if gdf.empty:
        raise ValueError(
            "No valid geometries remain after cleaning."
        )

    return gdf


# ============================================================
# EVALUATE ONE YEAR
# ============================================================

def evaluate_year(year, prediction_path, label_path):

    print()
    print("=" * 80)
    print(f"FRANCE {year}")
    print("=" * 80)

    print(f"Prediction: {prediction_path}")
    print(f"Reference:  {label_path}")

    # --------------------------------------------------------
    # Check prediction file
    # --------------------------------------------------------

    if not Path(prediction_path).exists():
        raise FileNotFoundError(
            f"\nPrediction raster not found:\n"
            f"{prediction_path}\n\n"
            f"Check the filename in YEARS."
        )

    # --------------------------------------------------------
    # 1. Read prediction raster
    # --------------------------------------------------------

    with rasterio.open(prediction_path) as src:

        prediction = src.read(1)

        transform = src.transform
        crs = src.crs

        height = src.height
        width = src.width

        nodata = src.nodata

        pixel_size_x = abs(transform.a)
        pixel_size_y = abs(transform.e)

        bounds = src.bounds

    print()
    print("PREDICTION RASTER")
    print("-" * 40)
    print(f"Size:       {width} x {height}")
    print(f"CRS:        {crs}")
    print(f"Resolution: {pixel_size_x} x {pixel_size_y}")
    print(f"NoData:     {nodata}")
    print(f"Bounds:     {bounds}")

    # --------------------------------------------------------
    # 2. Identify valid pixels
    # --------------------------------------------------------

    valid = np.isfinite(prediction)

    if nodata is not None:
        valid &= prediction != nodata

    valid_values = prediction[valid]

    if valid_values.size == 0:
        raise ValueError(
            "Prediction raster contains no valid pixels."
        )

    print(
        f"Prediction range: "
        f"{valid_values.min():.6f} - "
        f"{valid_values.max():.6f}"
    )

    # --------------------------------------------------------
    # 3. Convert prediction to binary mask
    # --------------------------------------------------------

    if (
        valid_values.min() >= 0
        and valid_values.max() <= 1
    ):

        prediction_mask = prediction >= THRESHOLD

        print(
            f"Prediction type: probability"
        )

        print(
            f"Threshold: {THRESHOLD}"
        )

    else:

        prediction_mask = prediction > 0

        print(
            "Prediction type: binary/non-probability"
        )

        print(
            "Using prediction > 0 as lavender."
        )

    prediction_mask &= valid

    # --------------------------------------------------------
    # 4. Load reference polygons
    # --------------------------------------------------------

    labels = load_reference_labels(
        label_path,
        crs
    )


    # --------------------------------------------------------
    # 5. Rasterize reference polygons
    # --------------------------------------------------------

    shapes = [
        (geom, 1)
        for geom in labels.geometry
        if geom is not None
        and not geom.is_empty
    ]

    print()
    print("RASTERIZING REFERENCE")
    print("-" * 40)

    reference_mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)

    # --------------------------------------------------------
    # 7. Pixel-level comparison
    # --------------------------------------------------------

    # Evaluate only valid prediction pixels.
    reference_eval = reference_mask[valid]
    prediction_eval = prediction_mask[valid]

    metrics = calculate_metrics(
        reference_eval,
        prediction_eval
    )

    # --------------------------------------------------------
    # 8. Calculate areas
    # --------------------------------------------------------

    pixel_area_m2 = pixel_size_x * pixel_size_y
    pixel_area_ha = pixel_area_m2 / 10000.0

    reference_pixels = int(reference_eval.sum())
    predicted_pixels = int(prediction_eval.sum())

    reference_area_ha = (
        reference_pixels * pixel_area_ha
    )

    predicted_area_ha = (
        predicted_pixels * pixel_area_ha
    )

    overlap_area_ha = (
        metrics["TP"] * pixel_area_ha
    )

    # --------------------------------------------------------
    # 9. Add information to results
    # --------------------------------------------------------

    metrics["Year"] = year

    metrics["Reference_fields"] = len(labels)

    metrics["Reference_pixels"] = reference_pixels
    metrics["Predicted_pixels"] = predicted_pixels

    metrics["Reference_area_ha"] = reference_area_ha
    metrics["Predicted_area_ha"] = predicted_area_ha
    metrics["Overlap_area_ha"] = overlap_area_ha

    # --------------------------------------------------------
    # 10. Print results
    # --------------------------------------------------------

    print()
    print("RESULTS")
    print("-" * 40)

    print(
        f"Reference fields: {len(labels)}"
    )

    print(
        f"Reference area:   "
        f"{reference_area_ha:,.2f} ha"
    )

    print(
        f"Predicted area:   "
        f"{predicted_area_ha:,.2f} ha"
    )

    print(
        f"Overlap area:     "
        f"{overlap_area_ha:,.2f} ha"
    )

    print()

    print(
        f"IoU:              "
        f"{metrics['IoU']:.4f}"
    )

    print(
        f"Dice / F1:        "
        f"{metrics['Dice']:.4f}"
    )

    print(
        f"Precision:        "
        f"{metrics['Precision']:.4f}"
    )

    print(
        f"Recall:           "
        f"{metrics['Recall']:.4f}"
    )

    print(
        f"Accuracy:         "
        f"{metrics['Accuracy']:.4f}"
    )

    print()

    print(
        f"TP: {metrics['TP']:,}"
    )

    print(
        f"FP: {metrics['FP']:,}"
    )

    print(
        f"FN: {metrics['FN']:,}"
    )

    print(
        f"TN: {metrics['TN']:,}"
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    results = []

    print("=" * 80)
    print("FRANCE TRANSFER VALIDATION")
    print("=" * 80)
    print(f"Threshold: {THRESHOLD}")
    print()

    # --------------------------------------------------------
    # Evaluate each year
    # --------------------------------------------------------

    for year, config in YEARS.items():

        result = evaluate_year(
            year=year,
            prediction_path=config["prediction"],
            label_path=config["labels"],
        )

        results.append(result)

    # --------------------------------------------------------
    # Create summary
    # --------------------------------------------------------

    df = pd.DataFrame(results)

    columns = [
        "Year",
        "Reference_fields",
        "Reference_area_ha",
        "Predicted_area_ha",
        "Overlap_area_ha",
        "IoU",
        "Dice",
        "Precision",
        "Recall",
        "Accuracy",
        "TP",
        "FP",
        "FN",
        "TN",
    ]

    df = df[columns]

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print()
    print("=" * 110)
    print("FRANCE TRANSFER VALIDATION SUMMARY")
    print("=" * 110)

    print()

    print(
        df.to_string(
            index=False,
            formatters={
                "Reference_area_ha":
                    "{:,.1f}".format,

                "Predicted_area_ha":
                    "{:,.1f}".format,

                "Overlap_area_ha":
                    "{:,.1f}".format,

                "IoU":
                    "{:.4f}".format,

                "Dice":
                    "{:.4f}".format,

                "Precision":
                    "{:.4f}".format,

                "Recall":
                    "{:.4f}".format,

                "Accuracy":
                    "{:.4f}".format,
            }
        )
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print()
    print("=" * 110)
    print(f"Results saved to:")
    print(OUTPUT_CSV)
    print("=" * 110)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

