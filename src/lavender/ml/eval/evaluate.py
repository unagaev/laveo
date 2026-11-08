import glob
import os

import torch
from torch.utils.data import DataLoader
from lavender.ml.dataset.dataset import LavenderDataset
from lavender.ml.models.resunet import ResUNet
from lavender.ml.eval.metrics import iou, dice


img_paths = sorted(glob.glob("data/cache/val/images/*.npy"))
mask_paths = sorted(glob.glob("data/cache/val/masks/*.npy"))

assert len(img_paths) == len(mask_paths), "Image/mask count mismatch"

assert len(img_paths) > 0, "No validation images found"

for img, mask in zip(img_paths, mask_paths):
    assert os.path.basename(img) == os.path.basename(mask), \
        f"Image/mask mismatch: {img}, {mask}"

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResUNet(in_channels=12).to(device)
    checkpoint = torch.load(
        "ml/models/resunet_lavender_best.pth",
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ds = LavenderDataset(
        img_paths,
        mask_paths,
        mode="val"
    )

    loader = DataLoader(ds, batch_size=4, shuffle=False)

    iou_sum, dice_sum, n = 0, 0, 0

    with torch.no_grad():
        for x, y in loader:

            x, y = x.to(device), y.to(device)

            pred = model(x)

            iou_sum += iou(pred, y).item()
            dice_sum += dice(pred, y).item()
            n += 1

    print("IoU:", iou_sum / n)
    print("Dice:", dice_sum / n)

if __name__ == "__main__":
    main()