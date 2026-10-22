import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# DICE LOSS (stable + standard)
# =========================
class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)

        dice = (2. * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        return 1 - dice.mean()


# =========================
# FOCAL BCE (IMBALANCE ROBUST)
# =========================
class FocalBCELoss(nn.Module):

    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        prob = torch.sigmoid(logits)

        pt = targets * prob + (1 - targets) * (1 - prob)

        loss = self.alpha * (1 - pt) ** self.gamma * bce

        return loss.mean()


# =========================
# COMBINED LOSS
# =========================
class CombinedLoss(nn.Module):

    def __init__(
        self,
        focal_weight=0.4,
        dice_weight=0.6,
        focal_alpha=0.8,
        focal_gamma=2.0
    ):

        super().__init__()

        self.focal = FocalBCELoss(
            alpha=focal_alpha,
            gamma=focal_gamma
        )

        self.dice = DiceLoss()

        self.focal_w = focal_weight
        self.dice_w = dice_weight

    def forward(self, logits, targets):

        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)

        return self.focal_w * focal_loss + self.dice_w * dice_loss