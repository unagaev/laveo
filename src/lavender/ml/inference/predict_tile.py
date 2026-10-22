import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import glob
from tqdm import tqdm

import torch
import rasterio
import numpy as np
import geopandas as gpd



from ..models.resunet import ResUNet
from ..eval.metrics import (
    iou,
    dice,
    precision,
    recall,
    accuracy
)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = (

    "ml/models/resunet_lavender_best.pth"
)

AOI_PATH = (
    "data/aoi/Dobrich_Province_Boundaries.geojson"
)

INPUT_DIR = (
    #"data/tiles/images/predicted"
    "data/tiles/predictions/images"
)

OUTPUT_DIR = (
    "data/output/predictions"
)

MASK_DIR = (
    "cache/val/masks"
)

PATCH_SIZE = 96

STRIDE = 96

IN_CHANNELS = 12

BASE_FILTERS = 32

OUT_CHANNELS = 1

THRESHOLD = 0.5


# ============================================================
# LOAD AOI
# ============================================================

aoi = gpd.read_file(
    AOI_PATH
)


# ============================================================
# MODEL
# ============================================================

def load_model(device):

    print("=" * 70)
    print("LOADING RESUNET")
    print("=" * 70)

    # --------------------------------------------------------
    # Create exactly the same architecture used for training
    # --------------------------------------------------------

    model = ResUNet(
        in_channels=IN_CHANNELS,
        base_filters=BASE_FILTERS,
        out_channels=OUT_CHANNELS
    )

    # --------------------------------------------------------
    # PyTorch 2.6+
    #
    # weights_only=False is required for checkpoints saved
    # as dictionaries containing numpy / optimizer metadata.
    #
    # Use this only for a checkpoint you trust.
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    # --------------------------------------------------------
    # Support BOTH checkpoint formats
    #
    # A) torch.save(model.state_dict(), path)
    #
    # B) torch.save({
    #        "model_state_dict": model.state_dict(),
    #        ...
    #    }, path)
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        print(
            "[INFO] Full checkpoint detected."
        )

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

        if "epoch" in checkpoint:

            print(
                f"[INFO] Epoch: "
                f"{checkpoint['epoch']}"
            )

        if "val_dice" in checkpoint:

            print(
                f"[INFO] Val Dice: "
                f"{checkpoint['val_dice']:.4f}"
            )

        if "val_iou" in checkpoint:

            print(
                f"[INFO] Val IoU: "
                f"{checkpoint['val_iou']:.4f}"
            )

    else:

        print(
            "[INFO] State-dict detected."
        )

        state_dict = checkpoint

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        f"[✓] Model loaded:"
    )

    print(
        f"    {MODEL_PATH}"
    )

    print(
        f"    Input channels: "
        f"{IN_CHANNELS}"
    )

    print(
        f"    Base filters: "
        f"{BASE_FILTERS}"
    )

    return model


# ============================================================
# PREPROCESS
# ============================================================

def prepare_patch(img):

    img = img.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Replace NaN / Inf
    # --------------------------------------------------------

    img = np.nan_to_num(
        img,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # STRICT 12-BAND CHECK
    # --------------------------------------------------------

    if img.shape[0] != IN_CHANNELS:

        raise ValueError(
            f"Expected "
            f"{IN_CHANNELS} bands, "
            f"got {img.shape[0]}"
        )

    # --------------------------------------------------------
    # Sentinel-2 reflectance scaling
    #
    # Training must use the same scaling.
    # --------------------------------------------------------

    img = img / 10000.0

    # --------------------------------------------------------
    # Keep same preprocessing as training
    # --------------------------------------------------------

    img = np.clip(
        img,
        -1,
        1
    )

    return img


# ============================================================
# PAD IMAGE
# ============================================================

def pad_image(
    img,
    patch_size
):

    c, h, w = img.shape

    pad_h = (
        patch_size
        - h % patch_size
    ) % patch_size

    pad_w = (
        patch_size
        - w % patch_size
    ) % patch_size

    if pad_h == 0 and pad_w == 0:

        return img, h, w

    img = np.pad(
        img,
        (
            (0, 0),
            (0, pad_h),
            (0, pad_w)
        ),
        mode="reflect"
    )

    return img, h, w


# ============================================================
# PREDICT ONE TILE
# ============================================================

def predict_single_tile(
    model,
    device,
    path,
    out_path
):

    print(
        f"[INFO] Reading: "
        f"{path}"
    )

    # --------------------------------------------------------
    # READ RASTER
    # --------------------------------------------------------

    with rasterio.open(path) as src:

        image = src.read()

        meta = src.meta.copy()

        crs = src.crs

        transform = src.transform

    print(
        f"[INFO] Input shape: "
        f"{image.shape}"
    )

    # --------------------------------------------------------
    # Check number of bands
    # --------------------------------------------------------

    if image.shape[0] != IN_CHANNELS:

        raise ValueError(
            f"\nExpected "
            f"{IN_CHANNELS} bands.\n"
            f"File: {path}\n"
            f"Got: {image.shape[0]}"
        )

    # --------------------------------------------------------
    # PAD
    # --------------------------------------------------------

    image, orig_h, orig_w = (
        pad_image(
            image,
            PATCH_SIZE
        )
    )

    c, h, w = image.shape

    print(
        f"[INFO] Padded shape: "
        f"{image.shape}"
    )

    # --------------------------------------------------------
    # Prediction accumulation
    # --------------------------------------------------------

    pred_sum = np.zeros(
        (h, w),
        dtype=np.float32
    )

    pred_count = np.zeros(
        (h, w),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # PATCH LOOP
    # --------------------------------------------------------

    for y in range(
        0,
        h,
        STRIDE
    ):

        for x in range(
            0,
            w,
            STRIDE
        ):

            patch = image[
                :,
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            # ------------------------------------------------
            # Safety
            # ------------------------------------------------

            if patch.shape != (
                IN_CHANNELS,
                PATCH_SIZE,
                PATCH_SIZE
            ):

                raise ValueError(
                    f"Unexpected patch shape: "
                    f"{patch.shape}"
                )

            # ------------------------------------------------
            # Preprocess
            # ------------------------------------------------

            patch = prepare_patch(
                patch
            )

            # ------------------------------------------------
            # Tensor
            # ------------------------------------------------

            tensor = (
                torch.from_numpy(
                    patch
                )
                .float()
                .unsqueeze(0)
                .to(device)
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            with torch.no_grad():

                logits = model(
                    tensor
                )

                probability = (
                    torch.sigmoid(
                        logits
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )

            # ------------------------------------------------
            # Accumulate
            # ------------------------------------------------

            pred_sum[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ] += probability

            pred_count[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ] += 1

    # ========================================================
    # AVERAGE OVERLAPPING PREDICTIONS
    # ========================================================

    pred = (
        pred_sum
        /
        np.maximum(
            pred_count,
            1
        )
    )

    # --------------------------------------------------------
    # Remove padding
    # --------------------------------------------------------

    pred = pred[
        :orig_h,
        :orig_w
    ]

    # --------------------------------------------------------
    # Binary mask
    # --------------------------------------------------------

    binary = (
        pred > THRESHOLD
    ).astype(
        np.uint8
    )

    # ========================================================
    # VALIDATION MASK
    # ========================================================

    mask_path = os.path.join(
        MASK_DIR,
        os.path.basename(path)
    )

    metrics = None

    if os.path.exists(
        mask_path
    ):

        print(
            f"[INFO] Ground truth found:"
        )

        print(
            f"       {mask_path}"
        )

        with rasterio.open(
            mask_path
        ) as src:

            gt = src.read(
                1
            ).astype(
                np.uint8
            )

        # ----------------------------------------------------
        # Check shape
        # ----------------------------------------------------

        if gt.shape != binary.shape:

            print(
                "[WARNING] Ground-truth "
                "shape does not match "
                "prediction."
            )

            print(
                f"GT:   {gt.shape}"
            )

            print(
                f"Pred: {binary.shape}"
            )

        else:

            dice_val = dice(
                binary,
                gt
            )

            iou_val = iou(
                binary,
                gt
            )

            precision_val = precision(
                binary,
                gt
            )

            recall_val = recall(
                binary,
                gt
            )

            accuracy_val = accuracy(
                binary,
                gt
            )

            metrics = (
                dice_val,
                iou_val,
                precision_val,
                recall_val,
                accuracy_val
            )

            print(
                f"Dice={dice_val:.4f}  "
                f"IoU={iou_val:.4f}  "
                f"Precision={precision_val:.4f}  "
                f"Recall={recall_val:.4f}  "
                f"Accuracy={accuracy_val:.4f}"
            )

    # ========================================================
    # SAVE PREDICTION
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Remove nodata from metadata
    # --------------------------------------------------------

    meta.pop(
        "nodata",
        None
    )

    meta.update(
        {
            "height":
                orig_h,

            "width":
                orig_w,

            "count":
                1,

            "dtype":
                "uint8",

            "compress":
                "lzw"
        }
    )

    with rasterio.open(
        out_path,
        "w",
        **meta
    ) as dst:

        dst.write(
            binary,
            1
        )

    print(
        f"[✓] Saved: "
        f"{out_path}"
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[INFO] Device: "
        f"{device}"
    )

    if torch.cuda.is_available():

        print(
            f"[INFO] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        device
    )

    # --------------------------------------------------------
    # Find tiles
    # --------------------------------------------------------

    tiles = sorted(
        glob.glob(
            os.path.join(
                INPUT_DIR,
                "*.tif"
            )
        )
    )

    print(
        f"\n[INFO] Found "
        f"{len(tiles)} tiles"
    )

    if len(tiles) == 0:

        raise RuntimeError(
            f"No TIFF files found in:\n"
            f"{INPUT_DIR}"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    all_dice = []

    all_iou = []

    all_precision = []

    all_recall = []

    all_accuracy = []

    # ========================================================
    # PROCESS TILES
    # ========================================================

    for i, tile_path in enumerate(
        tqdm(
            tiles,
            desc="Predicting tiles",
            unit="tile"
        )
    ):

        name = os.path.splitext(
            os.path.basename(tile_path)
        )[0]

        out_path = os.path.join(
            OUTPUT_DIR,
            f"{name}_prediction.tif"
        )

        metrics = predict_single_tile(
            model,
            device,
            tile_path,
            out_path
        )

        if metrics is not None:

            (
                dice_v,
                iou_v,
                prec_v,
                rec_v,
                acc_v
            ) = metrics

            all_dice.append(
                dice_v
            )

            all_iou.append(
                iou_v
            )

            all_precision.append(
                prec_v
            )

            all_recall.append(
                rec_v
            )

            all_accuracy.append(
                acc_v
            )

    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    if all_dice:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "OVERALL RESULTS"
        )

        print(
            "=" * 70
        )

        print(
            f"Tiles evaluated: "
            f"{len(all_dice)}"
        )

        print(
            f"Dice      : "
            f"{np.mean(all_dice):.4f}"
        )

        print(
            f"IoU       : "
            f"{np.mean(all_iou):.4f}"
        )

        print(
            f"Precision : "
            f"{np.mean(all_precision):.4f}"
        )

        print(
            f"Recall    : "
            f"{np.mean(all_recall):.4f}"
        )

        print(
            f"Accuracy  : "
            f"{np.mean(all_accuracy):.4f}"
        )

    else:

        print(
            "\n[INFO] No ground-truth "
            "masks were found."
        )

    print(
        "\nDONE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()