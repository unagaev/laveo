import os
import time
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from ..models.resunet import ResUNet

from ..training.losses import CombinedLoss

from ..eval.metrics import (
    iou,
    dice,
    precision,
    recall,
    accuracy
)

from src.lavender.utils.patches_tiles_all_years import (
    PATCH_SIZE,
    STRIDE
)


# ============================================================
# CONFIG
# ============================================================

# ------------------------------------------------------------
# MODEL SELECTION
# ------------------------------------------------------------
#
# Choose:
#
#     "unet"
#
# or:
#
#     "resunet"
#
MODEL_TYPE = "resunet"
#MODEL_TYPE = "unet"

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

TRAIN_IMG = "data/cache/train/images"
TRAIN_MASK = "data/cache/train/masks"

VAL_IMG = "data/cache/val/images"
VAL_MASK = "data/cache/val/masks"

TEST_IMG = "data/cache/test/images"
TEST_MASK = "data/cache/test/masks"


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

IN_CHANNELS = 12

OUT_CHANNELS = 1

BASE_FILTERS = 32


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------

NUM_EPOCHS = 50

BATCH_SIZE = 4

LR = 1e-4

NUM_WORKERS = 0

SEED = 42

PATIENCE = 8


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

MODEL_DIR = "ml/models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# MODEL OUTPUT PATH
# ============================================================

if MODEL_TYPE.lower() == "unet":

    BEST_MODEL_PATH = os.path.join(
        MODEL_DIR,
        "unet_lavender_best.pth"
    )

elif MODEL_TYPE.lower() == "resunet":

    BEST_MODEL_PATH = os.path.join(
        MODEL_DIR,
        "resunet_lavender_best.pth"
    )

else:

    raise ValueError(
        f"Unknown MODEL_TYPE: {MODEL_TYPE}\n"
        f"Use 'unet' or 'resunet'."
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# OPENMP
# ============================================================

# Windows sometimes has a conflict between
# libomp.dll and libiomp5md.dll.
#
# This avoids the crash you encountered.
#
# Ideally the environment should eventually be cleaned
# so that only one OpenMP runtime exists.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("LAVENDER SEGMENTATION TRAINING")
print("=" * 70)

print(
    f"Model:          {MODEL_TYPE.upper()}"
)

print(
    f"Device:         {DEVICE}"
)

print(
    f"Input channels: {IN_CHANNELS}"
)

print(
    f"Patch size:     {PATCH_SIZE} × {PATCH_SIZE}"
)

print(
    f"Batch size:     {BATCH_SIZE}"
)

print(
    f"Learning rate:  {LR}"
)

print(
    f"Epochs:         {NUM_EPOCHS}"
)

print(
    f"Best model:     {BEST_MODEL_PATH}"
)

if torch.cuda.is_available():

    print(
        f"GPU:            "
        f"{torch.cuda.get_device_name(0)}"
    )


# ============================================================
# DATASET
# ============================================================

class VineyardDataset(Dataset):

    def __init__(
        self,
        img_dir,
        mask_dir,
        mode="train",
        vineyard_ratio=0.7
    ):

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.mode = mode

        self.vineyard_ratio = vineyard_ratio

        # ----------------------------------------------------
        # Find image files
        # ----------------------------------------------------

        self.img_files = sorted(
            [
                f for f in os.listdir(img_dir)
                if f.endswith(".npy")
            ],
            key=lambda x:
                int(os.path.splitext(x)[0])
                if os.path.splitext(x)[0].isdigit()
                else x
        )

        if len(self.img_files) == 0:

            raise RuntimeError(
                f"No .npy images found in:\n"
                f"{img_dir}"
            )

        # ----------------------------------------------------
        # Match masks explicitly
        # ----------------------------------------------------

        self.samples = []

        for filename in self.img_files:

            img_path = os.path.join(
                img_dir,
                filename
            )

            mask_path = os.path.join(
                mask_dir,
                filename
            )

            if not os.path.exists(mask_path):

                raise FileNotFoundError(
                    f"\nMissing mask:\n"
                    f"{mask_path}\n"
                    f"for image:\n"
                    f"{img_path}"
                )

            self.samples.append(
                (
                    img_path,
                    mask_path
                )
            )

        # ----------------------------------------------------
        # Identify positive / empty samples
        # ----------------------------------------------------

        self.vineyard_idx = []

        self.empty_idx = []

        for i, (_, mask_path) in enumerate(
            self.samples
        ):

            mask = np.load(
                mask_path
            )

            mask = mask > 0

            if mask.sum() > 0:

                self.vineyard_idx.append(i)

            else:

                self.empty_idx.append(i)

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        print(
            f"\n[Dataset: {mode}]"
        )

        print(
            f"  Total:    {len(self.samples)}"
        )

        print(
            f"  Vineyard: {len(self.vineyard_idx)}"
        )

        print(
            f"  Empty:    {len(self.empty_idx)}"
        )

        if len(self.samples) > 0:

            print(
                f"  Positive: "
                f"{100 * len(self.vineyard_idx) / len(self.samples):.1f}%"
            )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, idx):

        # ====================================================
        # TRAIN SAMPLING
        # ====================================================

        if self.mode == "train":

            if len(self.vineyard_idx) == 0:

                idx = np.random.randint(
                    len(self.samples)
                )

            elif len(self.empty_idx) == 0:

                idx = np.random.choice(
                    self.vineyard_idx
                )

            elif np.random.rand() < self.vineyard_ratio:

                idx = np.random.choice(
                    self.vineyard_idx
                )

            else:

                idx = np.random.choice(
                    self.empty_idx
                )

        # ====================================================
        # FILES
        # ====================================================

        img_path, mask_path = self.samples[idx]

        # ====================================================
        # LOAD IMAGE
        # ====================================================

        img = np.load(
            img_path
        ).astype(
            np.float32
        )

        # ====================================================
        # LOAD MASK
        # ====================================================

        mask = np.load(
            mask_path
        ).astype(
            np.float32
        )

        # ====================================================
        # IMAGE SHAPE
        # ====================================================

        expected_shape = (
            IN_CHANNELS,
            PATCH_SIZE,
            PATCH_SIZE
        )

        if img.shape != expected_shape:

            raise ValueError(
                f"\nInvalid image shape\n"
                f"File: {img_path}\n"
                f"Expected: {expected_shape}\n"
                f"Got: {img.shape}"
            )

        # ====================================================
        # MASK SHAPE
        # ====================================================

        if mask.ndim == 2:

            mask = mask[None, :, :]

        elif mask.ndim == 3 and mask.shape[0] == 1:

            pass

        else:

            raise ValueError(
                f"\nInvalid mask shape\n"
                f"File: {mask_path}\n"
                f"Got: {mask.shape}"
            )

        expected_mask_shape = (
            1,
            PATCH_SIZE,
            PATCH_SIZE
        )

        if mask.shape != expected_mask_shape:

            raise ValueError(
                f"\nInvalid mask shape\n"
                f"File: {mask_path}\n"
                f"Expected: {expected_mask_shape}\n"
                f"Got: {mask.shape}"
            )

        # ====================================================
        # NUMERICAL CHECK
        # ====================================================

        if not np.isfinite(img).all():

            raise ValueError(
                f"\nNaN/Inf found in image:\n"
                f"{img_path}"
            )

        if not np.isfinite(mask).all():

            raise ValueError(
                f"\nNaN/Inf found in mask:\n"
                f"{mask_path}"
            )

        # ====================================================
        # BINARY MASK
        # ====================================================

        mask = (
            mask > 0
        ).astype(
            np.float32
        )

        # ====================================================
        # TENSORS
        # ====================================================

        img = torch.from_numpy(
            img
        ).float()

        mask = torch.from_numpy(
            mask
        ).float()

        return img, mask


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    device
):

    model.eval()

    ious = []

    dices = []

    precisions = []

    recalls = []

    accuracies = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            y = y.to(device)

            logits = model(x)

            ious.append(
                iou(
                    logits,
                    y
                ).item()
            )

            dices.append(
                dice(
                    logits,
                    y
                ).item()
            )

            precisions.append(
                precision(
                    logits,
                    y
                ).item()
            )

            recalls.append(
                recall(
                    logits,
                    y
                ).item()
            )

            accuracies.append(
                accuracy(
                    logits,
                    y
                ).item()
            )

    return {

        "iou": np.mean(ious),

        "dice": np.mean(dices),

        "precision": np.mean(precisions),

        "recall": np.mean(recalls),

        "accuracy": np.mean(accuracies)
    }


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    if MODEL_TYPE.lower() == "unet":

        model = UNet(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS
        )

    elif MODEL_TYPE.lower() == "resunet":

        model = ResUNet(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            base_filters=BASE_FILTERS
        )

    else:

        raise ValueError(
            f"Unknown model: {MODEL_TYPE}"
        )

    return model.to(
        DEVICE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # DATASETS
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    train_ds = VineyardDataset(
        TRAIN_IMG,
        TRAIN_MASK,
        mode="train",
        vineyard_ratio=0.7
    )

    val_ds = VineyardDataset(
        VAL_IMG,
        VAL_MASK,
        mode="val"
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    model = create_model()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Model:           {MODEL_TYPE.upper()}"
    )

    print(
        f"Parameters:      {parameter_count:,}"
    )

    print(
        f"Trainable:       {trainable_count:,}"
    )

    # ========================================================
    # MODEL SHAPE TEST
    # ========================================================

    print("\nTesting model forward pass...")

    test_x = torch.randn(
        2,
        IN_CHANNELS,
        PATCH_SIZE,
        PATCH_SIZE
    ).to(DEVICE)

    with torch.no_grad():

        test_y = model(
            test_x
        )

    print(
        f"Input:  {tuple(test_x.shape)}"
    )

    print(
        f"Output: {tuple(test_y.shape)}"
    )

    expected_output = (
        2,
        OUT_CHANNELS,
        PATCH_SIZE,
        PATCH_SIZE
    )

    if tuple(test_y.shape) != expected_output:

        raise RuntimeError(
            f"Unexpected model output shape.\n"
            f"Expected: {expected_output}\n"
            f"Got:      {tuple(test_y.shape)}"
        )

    print(
        "✓ Model shape test passed."
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = CombinedLoss()

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    # ========================================================
    # TRAINING STATE
    # ========================================================

    best_iou = -float("inf")

    no_improve = 0

    arr_loss = []

    arr_iou = []

    arr_dice = []

    # ========================================================
    # TRAINING
    # ========================================================

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(
        NUM_EPOCHS
    ):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        total_loss = 0.0

        for x, y in train_loader:

            x = x.to(
                DEVICE,
                non_blocking=True
            )

            y = y.to(
                DEVICE,
                non_blocking=True
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            pred = model(x)

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                pred,
                y
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

        total_loss /= len(
            train_loader
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        metrics = validate(
            model,
            val_loader,
            DEVICE
        )

        val_iou = metrics["iou"]

        val_dice = metrics["dice"]

        # ----------------------------------------------------
        # Store metrics
        # ----------------------------------------------------

        arr_loss.append(
            total_loss
        )

        arr_iou.append(
            val_iou
        )

        arr_dice.append(
            val_dice
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Epoch {epoch:03d}: "
            f"loss={total_loss:.4f} | "
            f"IoU={val_iou:.4f} | "
            f"Dice={val_dice:.4f} | "
            f"Prec={metrics['precision']:.4f} | "
            f"Rec={metrics['recall']:.4f} | "
            f"Acc={metrics['accuracy']:.4f}"
        )

        # ====================================================
        # PLOT
        # ====================================================

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            arr_loss,
            label="loss"
        )

        plt.plot(
            arr_iou,
            label="IoU"
        )

        plt.plot(
            arr_dice,
            label="Dice"
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Value"
        )

        plt.title(
            f"{MODEL_TYPE.upper()} training"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            f"metrics_{MODEL_TYPE}_{PATCH_SIZE}_{STRIDE}.png",
            dpi=150
        )

        plt.close()

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if val_iou > best_iou:

            best_iou = val_iou

            no_improve = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "epoch":
                        epoch,

                    "val_iou":
                        val_iou,

                    "val_dice":
                        val_dice,

                    "config": {
                        "model":
                            MODEL_TYPE,

                        "in_channels":
                            IN_CHANNELS,

                        "out_channels":
                            OUT_CHANNELS,

                        "base_filters":
                            BASE_FILTERS
                    }
                },
                BEST_MODEL_PATH
            )

            print(
                f"[✓] Best model saved "
                f"(IoU={best_iou:.4f})"
            )

        else:

            no_improve += 1

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if no_improve >= PATIENCE:

            print(
                f"[Early stopping] "
                f"No IoU improvement for "
                f"{PATIENCE} epochs."
            )

            break

    # ========================================================
    # FINISHED
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Model:       {MODEL_TYPE.upper()}"
    )

    print(
        f"Best IoU:    {best_iou:.4f}"
    )

    print(
        f"Model saved: {BEST_MODEL_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    main()