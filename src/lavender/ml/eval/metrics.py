import torch

THR = 0.5


# =========================
# COMMON BINARIZATION
# =========================
def _binarize(logits, thr=THR):
    probs = torch.sigmoid(logits)
    return (probs > thr).float()


# =========================
# IOU (batch-safe, correct)
# =========================
def iou(logits, target, thr=THR, eps=1e-6):

    preds = _binarize(logits, thr)

    preds = preds.view(preds.shape[0], -1)
    target = target.view(target.shape[0], -1)

    inter = (preds * target).sum(dim=1)
    union = preds.sum(dim=1) + target.sum(dim=1) - inter

    return ((inter + eps) / (union + eps)).mean()


# =========================
# DICE (F1)
# =========================
def dice(logits, target, thr=THR, eps=1e-6):

    preds = _binarize(logits, thr)

    preds = preds.view(preds.shape[0], -1)
    target = target.view(target.shape[0], -1)

    inter = (preds * target).sum(dim=1)

    dice = (2 * inter + eps) / (preds.sum(dim=1) + target.sum(dim=1) + eps)

    return dice.mean()


# =========================
# PRECISION
# =========================
def precision(logits, target, thr=THR, eps=1e-6):

    preds = _binarize(logits, thr)

    preds = preds.view(preds.shape[0], -1)
    target = target.view(target.shape[0], -1)

    tp = (preds * target).sum(dim=1)
    fp = (preds * (1 - target)).sum(dim=1)

    return ((tp + eps) / (tp + fp + eps)).mean()


# =========================
# RECALL
# =========================
def recall(logits, target, thr=THR, eps=1e-6):

    preds = _binarize(logits, thr)

    preds = preds.view(preds.shape[0], -1)
    target = target.view(target.shape[0], -1)

    tp = (preds * target).sum(dim=1)
    fn = (target * (1 - preds)).sum(dim=1)

    return ((tp + eps) / (tp + fn + eps)).mean()


# =========================
# ACCURACY
# =========================
def accuracy(logits, target, thr=THR):

    preds = _binarize(logits, thr)
    return (preds == target).float().mean()