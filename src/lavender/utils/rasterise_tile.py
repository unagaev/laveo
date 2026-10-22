import numpy as np
from pathlib import Path

import geopandas as gpd
import rasterio
import rasterio.features


# ============================================================
# RASTERISE MASK
# ============================================================

def rasterise_mask(image_path, label_gdf, out_path):

    print(f"\nRaster:")
    print(f"  {image_path}")

    with rasterio.open(image_path) as src:
        transform = src.transform
        height = src.height
        width = src.width
        crs = src.crs

        print(f"  Size: {width} × {height}")
        print(f"  CRS: {crs}")

    if crs is None:
        raise ValueError(f"Raster has no CRS: {image_path}")

    # --------------------------------------------------------
    # Reproject labels to raster CRS
    # --------------------------------------------------------

    labels = label_gdf.to_crs(crs)

    # --------------------------------------------------------
    # Remove empty and invalid geometries
    # --------------------------------------------------------

    labels = labels[
        labels.geometry.notna()
        & ~labels.geometry.is_empty
    ].copy()

    labels = labels[
        labels.geometry.is_valid
    ]

    print(f"  Label features: {len(labels):,}")

    # --------------------------------------------------------
    # Rasterise
    # --------------------------------------------------------

    shapes = [
        (geom, 1)
        for geom in labels.geometry
    ]

    mask = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    lavender_pixels = int(mask.sum())

    print(f"  Lavender pixels: {lavender_pixels:,}")

    if lavender_pixels == 0:
        print("  WARNING: No lavender pixels were rasterised.")

    # --------------------------------------------------------
    # Save mask
    # --------------------------------------------------------

    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "LZW",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mask, 1)

    print(f"  Mask saved: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Paths and labels for each dataset
    # --------------------------------------------------------

    datasets = {
        "training": {
            "image_dir": Path("data/tiles/training/images"),
            "mask_dir": Path("data/tiles/training/masks"),
            "label_files": {
                "2021": Path(
                    "data/labels/training/Train_Bg_Lavender fields-07-2021.geojson"
                ),
                "2023": Path(
                    "data/labels/training/Train_Bg_Lavender fields-07-2023.geojson"
                ),
                "2025": Path(
                    "data/labels/training/Train_Bg_Lavender fields-07-2025.geojson"
                ),
            },
        },
        "test": {
            "image_dir": Path("data/tiles/test/images"),
            "mask_dir": Path("data/tiles/test/masks"),
            "label_files": {
                "2021": Path(
                    "data/labels/test/Test_Bg_Lavender fields-07-2021.geojson"
                ),
                "2023": Path(
                    "data/labels/test/Test_Bg_Lavender fields-07-2023.geojson"
                ),
                "2025": Path(
                    "data/labels/test/Test_Bg_Lavender fields-07-2025.geojson"
                ),
            },
        },
    }

    print("=" * 70)
    print("LAVENDER MASK RASTERISATION")
    print("=" * 70)

    total_created = 0
    total_skipped = 0

    # --------------------------------------------------------
    # Process training and test independently
    # --------------------------------------------------------

    for dataset_name, paths in datasets.items():

        image_dir = paths["image_dir"]
        mask_dir = paths["mask_dir"]
        label_files = paths["label_files"]

        print("\n" + "=" * 70)
        print(f"PROCESSING: {dataset_name.upper()}")
        print("=" * 70)

        print(f"Image directory: {image_dir}")
        print(f"Mask directory:  {mask_dir}")

        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {image_dir}"
            )

        mask_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------
        # Load labels for this dataset
        # ----------------------------------------------------

        labels_by_year = {}

        for year, label_path in label_files.items():

            if not label_path.is_file():
                raise FileNotFoundError(
                    f"Label file not found: {label_path}"
                )

            gdf = gpd.read_file(label_path)

            if gdf.empty:
                raise ValueError(
                    f"No geometries found in {label_path}"
                )

            if gdf.crs is None:
                raise ValueError(
                    f"Label file has no CRS: {label_path}"
                )

            labels_by_year[year] = gdf

            print(
                f"Loaded {year}: {len(gdf):,} features "
                f"from {label_path.name}"
            )

        # ----------------------------------------------------
        # Find images
        # ----------------------------------------------------

        image_files = sorted(image_dir.glob("*.tif"))

        if not image_files:
            print(f"No TIFF images found in {image_dir}")
            continue

        print(f"\nFound {len(image_files)} images.")

        # ----------------------------------------------------
        # Rasterise masks
        # ----------------------------------------------------

        for image_path in image_files:

            filename = image_path.name

            # Detect year from image filename
            matched_year = next(
                (
                    year
                    for year in labels_by_year
                    if year in filename
                ),
                None,
            )

            if matched_year is None:
                print(f"\nSKIPPING: Cannot determine year: {filename}")
                total_skipped += 1
                continue

            mask_path = mask_dir / filename.replace(
                ".tif",
                "_mask.tif",
            )

            print(f"\nYear: {matched_year}")

            rasterise_mask(
                image_path=image_path,
                label_gdf=labels_by_year[matched_year],
                out_path=mask_path,
            )

            total_created += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Masks created: {total_created}")
    print(f"Images skipped: {total_skipped}")


if __name__ == "__main__":
    main()