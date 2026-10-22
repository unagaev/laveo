import torch
from torch.utils.data import DataLoader
from ml.dataset.dataset import VineyardDataset
from ml.models.unet import UNet
from ml.eval.metrics import iou, dice

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(in_channels=12).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    ds = VineyardDataset(
        "data/cache/val/images",
        "data/cache/val/masks",
        mode="val",
        augment=False
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