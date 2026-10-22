import os
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm


# =========================
# CONFIG
# =========================

#INPUT_DIR = Path("data/patches")
INPUT_DIR = Path("data/patches")
OUTPUT_DIR = Path("data/cache")

SPLITS = ["train", "val", "test"]

IMG_FOLDER = "images"
MASK_FOLDER = "masks"

NORM_SCALE = 10000.0


# =========================
# SAFETY CHECK
# =========================
def check_structure():

    print("\n[CHECK] Dataset structure:")

    for split in SPLITS:

        img_path = INPUT_DIR / split / IMG_FOLDER
        mask_path = INPUT_DIR / split / MASK_FOLDER

        print(f"  {split}:")
        print("    images exists:", img_path.exists())
        print("    masks exists:", mask_path.exists())


# =========================
# CREATE OUTPUT DIRS
# =========================
def create_dirs():

    for split in SPLITS:

        (OUTPUT_DIR / split / IMG_FOLDER).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT_DIR / split / MASK_FOLDER).mkdir(
            parents=True,
            exist_ok=True
        )


# =========================
# CONVERT SINGLE FILE
# =========================
def convert_pair(img_path, mask_path):

    with rasterio.open(img_path) as src:
        img = src.read().astype(np.float32)

    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.uint8)

    # clean NaNs
    img = np.nan_to_num(img)

    # normalize once
    img = img / NORM_SCALE
    
    img = np.clip(img, -1, 1)

    return img, mask


# =========================
# PROCESS SPLIT
# =========================
def process_split(split):

    img_dir = INPUT_DIR / split / IMG_FOLDER
    mask_dir = INPUT_DIR / split / MASK_FOLDER

    if not img_dir.exists() or not mask_dir.exists():
        print(f"\n[SKIP] {split} folder missing → NOT processed")
        return

    img_files = sorted(os.listdir(img_dir))

    print(f"\n[INFO] Processing {split}: {len(img_files)} files")

    for f in tqdm(img_files):

        img_path = img_dir / f
        mask_path = mask_dir / f

        if not mask_path.exists():
            print(f"[WARN] Missing mask for {f}")
            continue

        img, mask = convert_pair(img_path, mask_path)

        out_name = f.replace(".tif", ".npy")

        np.save(
            OUTPUT_DIR / split / IMG_FOLDER / out_name,
            img
        )

        np.save(
            OUTPUT_DIR / split / MASK_FOLDER / out_name,
            mask
        )


# =========================
# MAIN
# =========================
def main():

    print("Starting TIFF → NPY conversion")

    check_structure()

    create_dirs()

    total = 0

    for split in SPLITS:

        process_split(split)

    print("\nDONE → cached dataset ready at:", OUTPUT_DIR)


if __name__ == "__main__":
    main()