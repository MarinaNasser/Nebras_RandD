"""
metrics.py
Detection metrics for single-class (polyp) object detection: IoU-matching,
Average Precision at IoU=0.5 (COCO-style 101-point interpolation), and
precision/recall/F1 at a fixed score threshold - the metrics standard polyp
detection papers (including REAL-Colon's own benchmark) report.
"""
import numpy as np
import torch


def box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of [xmin,ymin,xmax,ymax] boxes. Returns [N,M]."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-7)


def match_predictions_to_gt(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    """
    Greedy matching, highest-confidence prediction first: a prediction is a
    True Positive if it hits an unmatched GT box at IoU >= iou_thresh, else a
    False Positive (each GT box can only be claimed once - extra detections on
    an already-matched polyp count as false positives, per standard practice).
    Returns (tp, fp) boolean arrays aligned with predictions sorted by
    descending score, plus the GT count (for recall).
    """
    n_gt = gt_boxes.shape[0]
    if pred_boxes.shape[0] == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool), n_gt

    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]

    tp = np.zeros(len(pred_boxes), dtype=bool)
    fp = np.zeros(len(pred_boxes), dtype=bool)

    if n_gt == 0:
        fp[:] = True
        return tp, fp, n_gt

    matched_gt = torch.zeros(n_gt, dtype=torch.bool)
    ious = box_iou_matrix(pred_boxes, gt_boxes)  # [n_pred, n_gt]
    for i in range(len(pred_boxes)):
        row = ious[i].clone()
        row[matched_gt] = -1  # already-claimed GT boxes can't be matched again
        best_iou, best_j = row.max(dim=0)
        if best_iou.item() >= iou_thresh:
            tp[i] = True
            matched_gt[best_j] = True
        else:
            fp[i] = True
    return tp, fp, n_gt


def compute_ap(all_tp, all_fp, all_scores, n_gt_total):
    """COCO-style 101-point interpolated Average Precision from concatenated
    per-image TP/FP flags and confidence scores across the whole eval set."""
    if n_gt_total == 0:
        return float("nan")
    if len(all_scores) == 0:
        return 0.0

    order = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[order]
    fp = np.array(all_fp)[order]

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / n_gt_total
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # standard "envelope": make precision non-increasing when read right-to-left
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    ap = 0.0
    for r in np.linspace(0, 1, 101):
        p = precision[recall >= r].max() if np.any(recall >= r) else 0.0
        ap += p
    return ap / 101


class DetectionEvaluator:
    """
    Accumulates predictions/targets batch by batch, then reports AP@0.5 plus
    precision/recall/F1 at a fixed score threshold.
    Usage: .reset() at epoch start, .update(preds, targets) per batch,
    .compute() at epoch end.
    """
    def __init__(self, iou_thresh=0.5, score_thresh_for_prf1=0.5):
        self.iou_thresh = iou_thresh
        self.score_thresh_for_prf1 = score_thresh_for_prf1
        self.reset()

    def reset(self):
        self._tp, self._fp, self._scores = [], [], []
        self._n_gt_total = 0
        self._prf1_tp = self._prf1_fp = self._prf1_fn = 0

    def update(self, preds, targets):
        for pred, target in zip(preds, targets):
            pred_boxes = pred["boxes"].detach().cpu()
            pred_scores = pred["scores"].detach().cpu()
            gt_boxes = target["boxes"].detach().cpu()

            # for AP: use every prediction the model returns (already NMS'd
            # inside the model at its configured score_thresh)
            tp, fp, n_gt = match_predictions_to_gt(pred_boxes, pred_scores, gt_boxes, self.iou_thresh)
            self._tp.extend(tp.tolist())
            self._fp.extend(fp.tolist())
            order = torch.argsort(pred_scores, descending=True)
            self._scores.extend(pred_scores[order].tolist())
            self._n_gt_total += n_gt

            # for precision/recall/F1: apply a fixed operating-point score threshold
            keep = pred_scores >= self.score_thresh_for_prf1
            tp2, fp2, _ = match_predictions_to_gt(pred_boxes[keep], pred_scores[keep], gt_boxes, self.iou_thresh)
            self._prf1_tp += int(tp2.sum())
            self._prf1_fp += int(fp2.sum())
            self._prf1_fn += n_gt - int(tp2.sum())

    def compute(self):
        ap50 = compute_ap(self._tp, self._fp, self._scores, self._n_gt_total)
        precision = self._prf1_tp / max(self._prf1_tp + self._prf1_fp, 1)
        recall = self._prf1_tp / max(self._prf1_tp + self._prf1_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        return {"AP50": ap50, "precision": precision, "recall": recall, "f1": f1}
