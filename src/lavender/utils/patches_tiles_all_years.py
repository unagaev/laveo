import numpy as np
import rasterio

from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from pathlib import Path
import csv
import shutil


# ============================================================
# CONFIG
# ============================================================

PATCH_SIZE = 96

# Currently using random sampling, not a regular grid.
# Kept here for reference.
STRIDE = 96

# Number of candidate patches requested per year
# This is NOT the number of lavender fields.
TARGET_TRAIN_PATCHES = {
    "2021": 2500,
    "2023": 300,
    "2025": 300,
}

TARGET_TEST_PATCHES = {
    "2021": 500,
    "2023": 300,
    "2025": 300,
}

# Train / validation split INSIDE THE TRAINING AREA
TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.20

# Minimum fraction of patch containing AOI.
# Since the raster already represents the training/test polygon,
# this is normally not needed as a separate AOI mask.
MIN_AOI_RATIO = 0.60

# Reject tiny positive fragments.
MIN_LAVENDER_PIXELS = 20

# Maximum empty/background patches relative to positives.
#
# Example:
# 100 positive patches -> maximum 60 empty patches.
EMPTY_RATIO = 0.60

# Maximum random attempts
MAX_TRIES_MULTIPLIER = 50

# Ensure patches do not overlap.
MIN_DIST = PATCH_SIZE

SEED = 42

np.random.seed(SEED)


# ============================================================
# PATHS
# ============================================================

TRAIN_IMAGE_DIR = Path("data/tiles/training/images")
TRAIN_MASK_DIR = Path("data/tiles/training/masks")

TEST_IMAGE_DIR = Path("data/tiles/test/images")
TEST_MASK_DIR = Path("data/tiles/test/masks")

OUT_DIR = Path("data/patches")


# ============================================================
# INPUT DATA
# ============================================================

TRAIN_DATA = {
    "2021": {
        "image": TRAIN_IMAGE_DIR /
        "Dobrich_2021-07-01_2021-07-03_Bulgaria_Dobrich_training_AOI.tif",
        "mask": TRAIN_MASK_DIR /
        "Dobrich_2021-07-01_2021-07-03_Bulgaria_Dobrich_training_AOI_mask.tif",
    },
    "2023": {
        "image": TRAIN_IMAGE_DIR /
        "Dobrich_2023-07-01_2023-07-03_Bulgaria_Dobrich_training_AOI.tif",
        "mask": TRAIN_MASK_DIR /
        "Dobrich_2023-07-01_2023-07-03_Bulgaria_Dobrich_training_AOI_mask.tif",
    },
    "2025": {
        "image": TRAIN_IMAGE_DIR /
        "Dobrich_2025-07-01_2025-07-03_Bulgaria_Dobrich_training_AOI.tif",
        "mask": TRAIN_MASK_DIR /
        "Dobrich_2025-07-01_2025-07-03_Bulgaria_Dobrich_training_AOI_mask.tif",
    },
}


TEST_DATA = {
    "2021": {
        "image": TEST_IMAGE_DIR /
        "Dobrich_2021-07-01_2021-07-03_Bulgaria_Dobrich_test_AOI.tif",
        "mask": TEST_MASK_DIR /
        "Dobrich_2021-07-01_2021-07-03_Bulgaria_Dobrich_test_AOI_mask.tif",
    },
    "2023": {
        "image": TEST_IMAGE_DIR /
        "Dobrich_2023-07-01_2023-07-03_Bulgaria_Dobrich_test_AOI.tif",
        "mask": TEST_MASK_DIR /
        "Dobrich_2023-07-01_2023-07-03_Bulgaria_Dobrich_test_AOI_mask.tif",
    },
    "2025": {
        "image": TEST_IMAGE_DIR /
        "Dobrich_2025-07-01_2025-07-03_Bulgaria_Dobrich_test_AOI.tif",
        "mask": TEST_MASK_DIR /
        "Dobrich_2025-07-01_2025-07-03_Bulgaria_Dobrich_test_AOI_mask.tif",
    },
}

# ============================================================
# CHECK PATHS
# ============================================================

def check_inputs():

    print("\nChecking input data...")

    for dataset_name, dataset in [
        ("TRAIN", TRAIN_DATA),
        ("TEST", TEST_DATA)
    ]:

        print(f"\n{dataset_name}")

        for year, paths in dataset.items():

            print(f"\n{year}")

            if not paths["image"].exists():

                raise FileNotFoundError(
                    f"Image not found:\n"
                    f"{paths['image']}"
                )

            if not paths["mask"].exists():

                raise FileNotFoundError(
                    f"Mask not found:\n"
                    f"{paths['mask']}"
                )

            print(
                f"  Image: {paths['image']}"
            )

            print(
                f"  Mask:  {paths['mask']}"
            )


# ============================================================
# OVERLAP
# ============================================================

def overlaps(
    x1,
    y1,
    x2,
    y2
):

    return (
        x1 < x2 + PATCH_SIZE
        and
        x1 + PATCH_SIZE > x2
        and
        y1 < y2 + PATCH_SIZE
        and
        y1 + PATCH_SIZE > y2
    )


# ============================================================
# SAVE PATCH
# ============================================================

def save_patch(
    image_patch,
    mask_patch,
    transform,
    crs,
    split,
    year,
    patch_id
):

    image_dir = (
        OUT_DIR /
        split /
        "images"
    )

    mask_dir = (
        OUT_DIR /
        split /
        "masks"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        f"{year}_{patch_id:07d}.tif"
    )

    image_path = (
        image_dir /
        filename
    )

    mask_path = (
        mask_dir /
        filename
    )

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    meta = {
        "driver": "GTiff",
        "height": PATCH_SIZE,
        "width": PATCH_SIZE,
        "count": image_patch.shape[0],
        "dtype": image_patch.dtype,
        "crs": crs,
        "transform": transform,
        "compress": "LZW"
    }

    with rasterio.open(
        image_path,
        "w",
        **meta
    ) as dst:

        dst.write(image_patch)

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    mask_meta = meta.copy()

    mask_meta.update({
        "count": 1,
        "dtype": "uint8"
    })

    with rasterio.open(
        mask_path,
        "w",
        **mask_meta
    ) as dst:

        dst.write(
            mask_patch.astype(
                np.uint8
            ),
            1
        )


# ============================================================
# SAMPLE PATCHES
# ============================================================

def sample_patches(
    image_path,
    mask_path,
    target_samples,
    dataset_name,
    year
):

    print("\n" + "=" * 70)
    print(
        f"{dataset_name.upper()} — {year}"
    )
    print("=" * 70)

    print(
        f"Image:\n{image_path}"
    )

    print(
        f"Mask:\n{mask_path}"
    )

    with rasterio.open(image_path) as src_img, \
         rasterio.open(mask_path) as src_mask:

        width = src_img.width
        height = src_img.height

        crs = src_img.crs
        transform = src_img.transform

        print(
            f"\nImage size: "
            f"{width} × {height}"
        )

        print(
            f"Bands: "
            f"{src_img.count}"
        )

        if (
            src_mask.width != width
            or
            src_mask.height != height
        ):

            raise ValueError(
                "Image and mask dimensions "
                "do not match."
            )

        # ----------------------------------------------------
        # Read mask
        # ----------------------------------------------------

        print("\nLoading mask...")

        full_mask = src_mask.read(1)

        mask_bin = (
            full_mask > 0
        ).astype(np.uint8)

        total_lavender = int(
            mask_bin.sum()
        )

        print(
            f"Total lavender pixels: "
            f"{total_lavender:,}"
        )

        # ----------------------------------------------------
        # Sampling
        # ----------------------------------------------------

        accepted = []

        tries = 0

        max_tries = (
            target_samples *
            MAX_TRIES_MULTIPLIER
        )

        print(
            f"\nTarget patches: "
            f"{target_samples}"
        )

        print(
            f"Maximum attempts: "
            f"{max_tries:,}"
        )

        # ----------------------------------------------------
        # Random sampling
        # ----------------------------------------------------

        while (
            len(accepted)
            <
            target_samples
            and
            tries
            <
            max_tries
        ):

            tries += 1

            x = np.random.randint(
                0,
                width - PATCH_SIZE + 1
            )

            y = np.random.randint(
                0,
                height - PATCH_SIZE + 1
            )

            # ------------------------------------------------
            # Mask patch
            # ------------------------------------------------

            patch_mask = mask_bin[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            if patch_mask.shape != (
                PATCH_SIZE,
                PATCH_SIZE
            ):

                continue

            lavender_pixels = int(
                patch_mask.sum()
            )

            # ------------------------------------------------
            # Reject tiny fragments
            # ------------------------------------------------

            if (
                lavender_pixels > 0
                and
                lavender_pixels
                <
                MIN_LAVENDER_PIXELS
            ):

                continue

            # ------------------------------------------------
            # Check overlap
            # ------------------------------------------------

            valid = True

            for (
                old_x,
                old_y,
                _
            ) in accepted:

                if overlaps(
                    x,
                    y,
                    old_x,
                    old_y
                ):

                    valid = False
                    break

            if not valid:
                continue

            # ------------------------------------------------
            # Read image
            # ------------------------------------------------

            window = Window(
                x,
                y,
                PATCH_SIZE,
                PATCH_SIZE
            )

            image_patch = src_img.read(
                window=window
            )

            if image_patch.shape[1:] != (
                PATCH_SIZE,
                PATCH_SIZE
            ):

                continue

            # ------------------------------------------------
            # Accept
            # ------------------------------------------------

            accepted.append(
                (
                    x,
                    y,
                    lavender_pixels
                )
            )

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        print("\n" + "-" * 70)
        print("INITIAL SAMPLING")
        print("-" * 70)

        print(
            f"Attempts: "
            f"{tries:,}"
        )

        print(
            f"Accepted: "
            f"{len(accepted)}"
        )

        positives = [
            a for a in accepted
            if a[2] > 0
        ]

        empties = [
            a for a in accepted
            if a[2] == 0
        ]

        print(
            f"Positive: "
            f"{len(positives)}"
        )

        print(
            f"Empty: "
            f"{len(empties)}"
        )

        if accepted:

            print(
                f"Positive rate: "
                f"{100 * len(positives) / len(accepted):.1f}%"
            )

        # ----------------------------------------------------
        # Empty balancing
        # ----------------------------------------------------

        np.random.shuffle(
            empties
        )

        max_empty = int(
            len(positives) *
            EMPTY_RATIO
        )

        empties = empties[
            :max_empty
        ]

        samples = (
            positives +
            empties
        )

        np.random.shuffle(
            samples
        )

        print("\n" + "-" * 70)
        print("FINAL SAMPLE SET")
        print("-" * 70)

        print(
            f"Positive: "
            f"{len(positives)}"
        )

        print(
            f"Empty: "
            f"{len(empties)}"
        )

        print(
            f"Total: "
            f"{len(samples)}"
        )

        if len(samples) == 0:

            raise RuntimeError(
                f"No patches generated "
                f"for {dataset_name} {year}"
            )

        return samples, crs, transform


# ============================================================
# SAVE SAMPLES
# ============================================================

def save_samples(
    samples,
    image_path,
    split,
    year,
    patch_counter
):

    with rasterio.open(
        image_path
    ) as src:

        for (
            x,
            y,
            lavender_pixels
        ) in samples:

            window = Window(
                x,
                y,
                PATCH_SIZE,
                PATCH_SIZE
            )

            image_patch = src.read(
                window=window
            )

            mask_dir = (
                TEST_MASK_DIR
                if split == "test"
                else TRAIN_MASK_DIR
            )
            
            mask_path = (
                mask_dir /
                image_path.name.replace(".tif", "_mask.tif")
            )

            with rasterio.open(
                mask_path
            ) as mask_src:

                mask_patch = mask_src.read(
                    1,
                    window=window
                )

            mask_bin = (
                mask_patch > 0
            ).astype(
                np.uint8
            )

            patch_transform = (
                window_transform(
                    window,
                    src.transform
                )
            )

            save_patch(
                image_patch=image_patch,
                mask_patch=mask_bin,
                transform=patch_transform,
                crs=src.crs,
                split=split,
                year=year,
                patch_id=patch_counter
            )

            patch_counter += 1

    return patch_counter


# ============================================================
# MANIFEST
# ============================================================

def create_manifest():

    manifest_path = (
        OUT_DIR /
        "manifest.csv"
    )

    rows = []

    for split in [
        "train",
        "val",
        "test"
    ]:

        image_dir = (
            OUT_DIR /
            split /
            "images"
        )

        mask_dir = (
            OUT_DIR /
            split /
            "masks"
        )

        if not image_dir.exists():
            continue

        for image_path in sorted(
            image_dir.glob("*.tif")
        ):

            mask_path = (
                mask_dir /
                image_path.name
            )

            name = image_path.stem

            year = name.split("_")[0]

            rows.append({
                "split": split,
                "year": year,
                "image": str(image_path),
                "mask": str(mask_path)
            })

    with open(
        manifest_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "year",
                "image",
                "mask"
            ]
        )

        writer.writeheader()

        writer.writerows(rows)

    print(
        f"\nManifest saved:\n"
        f"{manifest_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MULTI-YEAR LAVENDER PATCH EXTRACTION")
    print("=" * 70)

    print(
        f"Patch size: "
        f"{PATCH_SIZE}"
    )

    print(
        "Training years: 2021, 2023, 2025"
    )

    print(
        "Independent test years: 2021, 2023, 2025"
    )

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    check_inputs()

    # --------------------------------------------------------
    # Remove old patches
    # --------------------------------------------------------

    if OUT_DIR.exists():

        print(
            f"\nRemoving existing patches:\n"
            f"{OUT_DIR}"
        )

        shutil.rmtree(
            OUT_DIR
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # TRAINING AREA
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CREATING TRAIN / VALIDATION DATA")
    print("=" * 70)

    train_counter = 0
    val_counter = 0

    for year in [
        "2021",
        "2023",
        "2025"
    ]:

        data = TRAIN_DATA[year]

        samples, _, _ = sample_patches(
            image_path=data["image"],
            mask_path=data["mask"],
            target_samples=TARGET_TRAIN_PATCHES[year],
            dataset_name="training area",
            year=year
        )

        # ----------------------------------------------------
        # Spatially independent train/validation split
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # We split the already spatially separated sampled
        # patches. Because patches themselves don't overlap,
        # this is substantially safer than randomly splitting
        # pixels.
        #
        # ----------------------------------------------------

        np.random.shuffle(
            samples
        )

        n_train = int(
            len(samples) *
            TRAIN_FRACTION
        )

        train_samples = (
            samples[:n_train]
        )

        val_samples = (
            samples[n_train:]
        )

        print("\nTRAIN / VAL SPLIT")

        print(
            f"{year} train: "
            f"{len(train_samples)}"
        )

        print(
            f"{year} val:   "
            f"{len(val_samples)}"
        )

        train_counter = save_samples(
            samples=train_samples,
            image_path=data["image"],
            split="train",
            year=year,
            patch_counter=train_counter
        )

        val_counter = save_samples(
            samples=val_samples,
            image_path=data["image"],
            split="val",
            year=year,
            patch_counter=val_counter
        )

    # ========================================================
    # INDEPENDENT TEST
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CREATING INDEPENDENT TEST DATA")
    print("=" * 70)

    test_counter = 0

    for year in [
        "2021",
        "2023",
        "2025"
    ]:

        data = TEST_DATA[year]

        samples, _, _ = sample_patches(
            image_path=data["image"],
            mask_path=data["mask"],
            target_samples=TARGET_TEST_PATCHES[year],
            dataset_name="independent test area",
            year=year
        )

        print(
            f"\nSaving independent "
            f"test patches for {year}"
        )

        test_counter = save_samples(
            samples=samples,
            image_path=data["image"],
            split="test",
            year=year,
            patch_counter=test_counter
        )

    # ========================================================
    # MANIFEST
    # ========================================================

    create_manifest()

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PATCHING COMPLETE")
    print("=" * 70)

    print(
        f"Training patches: "
        f"{train_counter}"
    )

    print(
        f"Validation patches: "
        f"{val_counter}"
    )

    print(
        f"Independent test patches: "
        f"{test_counter}"
    )

    print("\nDataset structure:")

    print(
        """
data/patches_multiyear/

    train/
        images/
        masks/

    val/
        images/
        masks/

    test/
        images/
        masks/

    manifest.csv
"""
    )

    print(
        "IMPORTANT:"
    )

    print(
        "The independent test area is never "
        "used for training or validation."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()