"""
eval_object_metrics.py
Evaluates an existing checkpoint using Object-Level (Lesion Hit/Coverage)
Precision and Recall, saving qualitative visual samples of False Alarms (FP)
and Misses (FN).
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from dataset import PolypDataset, build_stratified_splits, get_val_transforms
from model import build_model

# Visualization Colors (BGR)
GREEN = (0, 255, 0)   # Ground Truth
RED = (0, 0, 255)     # Prediction


def get_connected_components(binary_mask):
    """
    Finds all isolated polyp blobs in a binary mask (H, W).
    Preserves all connected components without dropping small blobs.
    """
    mask_u8 = (binary_mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_u8)
    
    components = []
    for label_id in range(1, num_labels):
        comp_mask = (labels == label_id).astype(np.float32)
        components.append(comp_mask)
    return components


def compute_object_level_counts(pred_mask, gt_mask, min_hit_overlap=0.05):
    """
    Hit-based polyp detection:
    - If a prediction overlaps >= min_hit_overlap of a GT polyp, it's a TP (no over-segmentation penalty).
    - If a prediction is on normal tissue with no GT overlap, it's a False Alarm (FP).
    - If a GT polyp is never touched, it's a Miss (FN).
    """
    pred_components = get_connected_components(pred_mask)
    gt_components = get_connected_components(gt_mask)

    tp = 0
    fp = 0
    matched_gt = set()

    for p_comp in pred_components:
        best_coverage = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_comp in enumerate(gt_components):
            inter = np.logical_and(p_comp, gt_comp).sum()
            gt_area = gt_comp.sum()
            
            coverage = inter / (gt_area + 1e-7)
            if coverage > best_coverage:
                best_coverage = coverage
                best_gt_idx = gt_idx

        if best_coverage >= min_hit_overlap:
            if best_gt_idx not in matched_gt:
                tp += 1
                matched_gt.add(best_gt_idx)
            # Duplicate overlapping regions on an already-detected polyp are not penalized as FP
        else:
            # Prediction on background with zero/insufficient polyp overlap
            fp += 1

    fn = len(gt_components) - len(matched_gt)
    return tp, fp, fn


def create_error_visual(img_tensor, pred_mask, gt_mask, title=""):
    """
    Creates a 3-panel visualization: Original Image | Ground Truth (Green) | Prediction (Red).
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Panel 1: Original Image
    panel_orig = img_bgr.copy()
    cv2.putText(panel_orig, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Panel 2: Ground Truth overlay (Green)
    panel_gt = img_bgr.copy()
    gt_contours, _ = cv2.findContours((gt_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel_gt, gt_contours, -1, GREEN, 2)
    cv2.putText(panel_gt, "GT (Green)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

    # Panel 3: Prediction overlay (Red)
    panel_pred = img_bgr.copy()
    pred_contours, _ = cv2.findContours((pred_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel_pred, pred_contours, -1, RED, 2)
    cv2.putText(panel_pred, f"Pred (Red) - {title}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

    combined = np.hstack([panel_orig, panel_gt, panel_pred])
    return combined


def evaluate_and_save_samples(model, loader, device, save_dir, min_hit_overlap=0.05, max_samples_per_error=4):
    model.eval()
    per_ds = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    saved_counts = defaultdict(lambda: {"fp": 0, "fn": 0})
    total_tp, total_fp, total_fn = 0, 0, 0

    fp_dir = save_dir / "fp_samples"
    fn_dir = save_dir / "fn_samples"
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for images, masks, ds_names in tqdm(loader, desc="Evaluating Hit-Based Metrics"):
            images_gpu = images.to(device)
            logits = model(images_gpu)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            gts = masks.numpy()

            for i in range(len(ds_names)):
                p_mask = preds[i, 0]
                g_mask = gts[i, 0]
                ds_name = ds_names[i]

                tp, fp, fn = compute_object_level_counts(p_mask, g_mask, min_hit_overlap=min_hit_overlap)
                per_ds[ds_name]["tp"] += tp
                per_ds[ds_name]["fp"] += fp
                per_ds[ds_name]["fn"] += fn

                total_tp += tp
                total_fp += fp
                total_fn += fn

                # Save true False Alarms (blobs hallucinated on clear background)
                if fp > 0 and saved_counts[ds_name]["fp"] < max_samples_per_error:
                    saved_counts[ds_name]["fp"] += 1
                    idx = saved_counts[ds_name]["fp"]
                    visual = create_error_visual(images[i], p_mask, g_mask, title=f"FP ({ds_name})")
                    cv2.imwrite(str(fp_dir / f"{ds_name}_fp_{idx}.png"), visual)

                # Save true Misses (polyp never touched by prediction)
                if fn > 0 and saved_counts[ds_name]["fn"] < max_samples_per_error:
                    saved_counts[ds_name]["fn"] += 1
                    idx = saved_counts[ds_name]["fn"]
                    visual = create_error_visual(images[i], p_mask, g_mask, title=f"FN ({ds_name})")
                    cv2.imwrite(str(fn_dir / f"{ds_name}_fn_{idx}.png"), visual)

    breakdown = {}
    for ds_name, counts in per_ds.items():
        prec = counts["tp"] / (counts["tp"] + counts["fp"] + 1e-7)
        rec = counts["tp"] / (counts["tp"] + counts["fn"] + 1e-7)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-7)
        breakdown[ds_name] = {
            "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"],
            "precision": float(prec), "recall": float(rec), "f1": float(f1)
        }

    overall_prec = total_tp / (total_tp + total_fp + 1e-7)
    overall_rec = total_tp / (total_tp + total_fn + 1e-7)
    overall_f1 = 2 * (overall_prec * overall_rec) / (overall_prec + overall_rec + 1e-7)

    overall = {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": float(overall_prec), "recall": float(overall_rec), "f1": float(overall_f1)
    }

    return overall, breakdown


def main(config_path, checkpoint_path=None, min_hit_overlap=0.05, max_samples=4):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = checkpoint_path or (Path(cfg["checkpoint_dir"]) / "best_model.pt")
    eval_save_dir = Path(cfg["output_dir"]) / "object_eval"
    eval_save_dir.mkdir(parents=True, exist_ok=True)

    # Load test dataset
    _, _, test_pairs = build_stratified_splits(cfg)
    test_loader = DataLoader(
        PolypDataset(test_pairs, transform=get_val_transforms(cfg["image_size"])),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"]
    )

    # Load model
    model = build_model(
        cfg.get("model_variant", "modified"),
        num_classes=cfg["num_classes"],
        pretrained=False,
        segformer_size=cfg.get("segformer_size", "b3")
    ).to(device)

    print(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    overall, per_ds = evaluate_and_save_samples(
        model, test_loader, device, save_dir=eval_save_dir,
        min_hit_overlap=min_hit_overlap, max_samples_per_error=max_samples
    )

    print("\n" + "=" * 80)
    print(f"LESION-LEVEL HIT EVALUATION (Min GT Coverage: {min_hit_overlap * 100:.0f}%)")
    print("=" * 80)
    print(f"{'DATASET':<22}{'TP':<6}{'FP':<6}{'FN':<6}{'PRECISION':<12}{'RECALL':<12}{'F1-SCORE':<10}")
    print("-" * 80)
    for ds_name, m in sorted(per_ds.items()):
        print(f"{ds_name:<22}{m['tp']:<6}{m['fp']:<6}{m['fn']:<6}{m['precision']:<12.4f}{m['recall']:<12.4f}{m['f1']:<10.4f}")
    print("-" * 80)
    print(f"{'OVERALL TEST SET':<22}{overall['tp']:<6}{overall['fp']:<6}{overall['fn']:<6}{overall['precision']:<12.4f}{overall['recall']:<12.4f}{overall['f1']:<10.4f}")
    print("=" * 80)

    out_file = eval_save_dir / f"hit_metrics_coverage_{min_hit_overlap}.json"
    with open(out_file, "w") as f:
        json.dump({"overall": overall, "per_dataset": per_ds}, f, indent=2)
        
    print(f"\nSaved metrics to: {out_file}")
    print(f"Saved False Alarm (FP) images to: {eval_save_dir / 'fp_samples'}")
    print(f"Saved Miss (FN) images to: {eval_save_dir / 'fn_samples'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--min_hit_overlap", type=float, default=0.05, 
                        help="Fraction of GT polyp area that must be intersected to count as a hit (default: 0.05 / 5%)")
    parser.add_argument("--samples", type=int, default=4, help="Number of FP and FN samples to save per dataset")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.min_hit_overlap, args.samples)