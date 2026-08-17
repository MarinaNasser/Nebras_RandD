"""
metrics.py
BCE + Dice combo loss, and Dice/IoU metrics standard in polyp-segmentation papers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coeff(pred_probs, target, eps=1e-7):
    pred_flat = pred_probs.reshape(pred_probs.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def iou_score(pred_probs, target, threshold=0.5, eps=1e-7):
    pred_bin = (pred_probs > threshold).float()
    pred_flat = pred_bin.reshape(pred_bin.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def pixel_accuracy(pred_probs, target, threshold=0.5):
    """
    Fraction of pixels classified correctly (polyp vs background), thresholded at 0.5.
    Reported alongside Dice/IoU since papers commonly cite it too, but it's a weaker
    metric here: polyps typically cover a small fraction of each frame, so a model
    that predicts "no polyp" everywhere already scores 90%+ on this metric without
    being useful. Dice/IoU are the metrics that actually reflect segmentation quality
    given that class imbalance - treat accuracy as a secondary sanity-check number,
    not the headline result.
    """
    pred_bin = (pred_probs > threshold).float()
    correct = (pred_bin == target).float()
    return correct.reshape(correct.size(0), -1).mean(dim=1).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        bce_loss = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dice_loss = 1 - dice_coeff(probs, target)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
