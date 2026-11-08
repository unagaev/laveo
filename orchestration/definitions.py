import os
import sys
import subprocess
from pathlib import Path
import time

from dagster import (asset, Definitions, MetadataValue)


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

INPUT_DIR = DATA_DIR / "tiles" / "predictions" / "images"

PREDICTION_DIR = DATA_DIR / "outputs" / "predictions"

EXPECTED_YEARS = (2021, 2023, 2025)


# ============================================================
# ENVIRONMENT
# ============================================================

def get_environment():

    env = os.environ.copy()

    src_path = str(PROJECT_ROOT / "src")

    existing = env.get("PYTHONPATH", "")

    env["PYTHONPATH"] = (
        src_path
        if not existing
        else src_path + os.pathsep + existing
    )

    return env

# ============================================================
# ASSET 1: DOWNLOAD TRAINING AND PREDICTION TILES
# ============================================================

start = time.time()

@asset
def download_training_test_tiles():
    print('[INFO]Downloading training tiles')
    subprocess.run([sys.executable, '-m', 'lavender.ingestion.download_training_test_tiles'], cwd=PROJECT_ROOT, env=get_environment(), check=True)
    print('[INFO]Training tile download complete')
    return True

@asset 
def download_prediction_tiles():
    print('Downloading prediction tiles')
    subprocess.run([sys.executable, '-m', 'lavender.ingestion.download_prediction_tiles'], cwd=PROJECT_ROOT, env=get_environment(), check=True)
    print('Prediction tile download complete')
    return True
    
# ============================================================
# ASSET 2: PREPARING TRAINING AND PREDICTION TILES
# ============================================================
@asset 
def rasterise_training_tiles(download_training_test_tiles):
    subprocess.run([sys.executable, '-m', 'lavender.utils.rasterise_tile'], cwd=PROJECT_ROOT, env=get_environment(), check=True)

@asset 
def generate_patches(rasterise_training_tiles):
    subprocess.run([sys.executable, '-m', 'lavender.utils.patches_tiles_all_years'], cwd=PROJECT_ROOT, env=get_environment(), check=True)
@asset 
def convert_patches(generate_patches):
    subprocess.run([sys.executable, '-m', 'lavender.utils.convert_patches_to_npy_multiyear'], cwd=PROJECT_ROOT, env=get_environment(), check=True)

# ============================================================
# ASSET 3: INPUT TILES
# ============================================================

@asset
def input_tiles(download_prediction_tiles):

    tiles = sorted(
        INPUT_DIR.glob("*.tif")
    )

    if not tiles:
        raise RuntimeError(
            f"No input TIFF files found: {INPUT_DIR}"
        )

    found_years = {
        year
        for tile in tiles
        for year in EXPECTED_YEARS
        if str(year) in tile.name
    }

    missing_years = (
        set(EXPECTED_YEARS) - found_years
    )

    if missing_years:
        raise RuntimeError(
            f"Missing input years: {sorted(missing_years)}"
        )

    paths = [str(tile) for tile in tiles]

    return paths

# ============================================================
# ASSET 4: TRAIN AND EVALUATE MODEL 
# ============================================================
@asset
def train_model(convert_patches):
    print('[INFO] Training ResUnet')
    subprocess.run([sys.executable, '-m', 'lavender.ml.training.train'], cwd=PROJECT_ROOT, env=get_environment(), check=True)
    
    return str(PROJECT_ROOT / 'ml' / 'models' / 'resunet_lavender_best.pth')

@asset
def evaluate_model(train_model):
    print('Evaluating trained model')
    subprocess.run([sys.executable, '-m', 'lavender.ml.eval.evaluate'], cwd=PROJECT_ROOT, env=get_environment(), check=True)
    
    return True


# ============================================================
# ASSET 5: FULL-TILE PREDICTIONS
# ============================================================

@asset
def predictions(input_tiles, evaluate_model):

    print("[INFO] Starting full-tile inference")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "lavender.ml.inference.predict_tile"
        ],
        cwd=PROJECT_ROOT,
        env=get_environment(),
        check=True
    )

    expected_outputs = [
        PREDICTION_DIR / (
            Path(path).stem + "_prediction.tif"
        )
        for path in input_tiles
    ]

    missing = [
        path
        for path in expected_outputs
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            f"Missing prediction outputs: {missing}"
        )

    print(
        f"[INFO] Generated {len(expected_outputs)} predictions"
    )

    return [
        str(path)
        for path in expected_outputs
    ]


# ============================================================
# ASSET 6: POSTPROCESSING
# ============================================================

@asset
def postprocessed_outputs(predictions):

    print("[INFO] Starting postprocessing")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "lavender.postprocessing.postprocessing"
        ],
        cwd=PROJECT_ROOT,
        env=get_environment(),
        check=True
    )

    print("[INFO] Postprocessing completed")

    return "Postprocessing completed"


# ============================================================
# DAGSTER DEFINITIONS
# ============================================================

defs = Definitions(
    assets=[
        download_training_test_tiles,
        download_prediction_tiles,
		rasterise_training_tiles,
        generate_patches,
        convert_patches,		
        input_tiles,
        train_model,
        evaluate_model,
        predictions,
        postprocessed_outputs
    ]
)
