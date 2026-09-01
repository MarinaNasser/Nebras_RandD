"""
inference_demo.py
Loads the best trained checkpoint and runs it on a folder of images, then
annotates each image based on the model's confidence there's a polyp:

  - confidence < 85%  -> draws a red circle in the top-left corner (flagged
                          "uncertain / needs manual review" - no box drawn,
                          since the mask isn't trusted enough to localize
                          anything at that confidence level)
  - confidence >= 85% -> draws a green bounding box around the predicted
                          polyp region (largest connected component of the
                          thresholded probability mask)

"Confidence" is a single per-image number derived from the per-pixel sigmoid
probability map: the mean of the top TOP_PIXEL_FRACTION most confident pixels,
not just the single max pixel - one bright outlier pixel would otherwise
always read as ~100% confident even on a genuinely polyp-free frame.

Usage:
    python inference_demo.py --config config.yaml --input_dir "path/to/images"
    python inference_demo.py --config config.yaml                # defaults to the held-out test set
    python inference_demo.py --config config.yaml --threshold 0.9 --n 20
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from dataset import build_cross_dataset_splits
from model import build_model

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

TOP_PIXEL_FRACTION = 0.01   # top 1% of pixels by predicted probability
MIN_TOP_PIXELS = 20         # floor, so small masks still get a stable estimate

RED = (0, 0, 255)    # BGR for cv2
GREEN = (0, 255, 0)


def preprocess(image_bgr, image_size):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    normed = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(normed.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def image_confidence(prob_map):
    """Mean of the top TOP_PIXEL_FRACTION most confident pixels in the
    predicted probability map - a more stable 'is there a polyp' signal
    than a single max pixel."""
    flat = prob_map.flatten()
    k = max(MIN_TOP_PIXELS, int(len(flat) * TOP_PIXEL_FRACTION))
    k = min(k, len(flat))
    top_k = np.partition(flat, -k)[-k:]
    return float(top_k.mean())


def largest_component_bbox(binary_mask):
    """Returns (x, y, w, h) of the largest connected component in a binary
    mask, or None if the mask is empty."""
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)  # x, y, w, h


def annotate(image_bgr, confidence, threshold, bbox=None):
    """Draws either a red 'uncertain' circle (top-left) or a green box around
    the predicted polyp, plus a confidence label, on a copy of the original-
    resolution BGR image."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    label = f"{confidence*100:.1f}%"

    confident_and_localized = (confidence >= threshold) and (bbox is not None)

    if confident_and_localized:
        x, y, bw, bh = bbox
        cv2.rectangle(out, (x, y), (x + bw, y + bh), GREEN, 2)
        cv2.putText(out, f"polyp {label}", (x, max(15, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
    else:
        # low confidence, OR confidence >= threshold but the thresholded mask
        # came up empty (can happen right at the boundary) - flag rather than
        # draw a box around nothing
        radius = max(10, int(min(h, w) * 0.04))
        center = (radius + 10, radius + 10)
        cv2.circle(out, center, radius, RED, thickness=-1)
        cv2.putText(out, f"uncertain {label}", (center[0] + radius + 8, center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
    return out


def main(cfg_path, input_dir, threshold, n_max):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.get("model_variant", "modified"), num_classes=cfg["num_classes"],
                         pretrained=False, segformer_size=cfg.get("segformer_size", "b0")).to(device)
    ckpt_path = Path(cfg["checkpoint_dir"]) / "best_model.pt"
    # weights_only=False: this is our own checkpoint (a plain state_dict of tensors),
    # but PyTorch 2.6's stricter default can still reject it depending on how it was
    # saved/what torch version wrote it - safe to relax since we trust the source.
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    print(f"Loaded best checkpoint from {ckpt_path}")

    if input_dir:
        img_paths = sorted([p for p in Path(input_dir).iterdir()
                             if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")])
        print(f"Running on {len(img_paths)} image(s) from {input_dir}")
    else:
        _, _, test_pairs = build_cross_dataset_splits(cfg)
        img_paths = [Path(p) for p, _ in test_pairs]
        print(f"No --input_dir given, defaulting to held-out test set "
              f"({cfg['held_out_dataset']}): {len(img_paths)} image(s)")

    if n_max:
        img_paths = img_paths[:n_max]

    out_dir = Path(cfg["output_dir"]) / "inference_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_size = cfg["image_size"]
    results = []

    with torch.no_grad():
        for img_path in img_paths:
            image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"[WARN] Could not read {img_path}, skipping.")
                continue
            orig_h, orig_w = image_bgr.shape[:2]

            tensor = preprocess(image_bgr, image_size).to(device)
            logits = model(tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (image_size, image_size)

            conf = image_confidence(probs)

            probs_full = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            binary_mask = (probs_full > 0.5).astype(np.uint8)
            bbox = largest_component_bbox(binary_mask) if conf >= threshold else None

            annotated = annotate(image_bgr, conf, threshold, bbox)
            out_path = out_dir / img_path.name
            cv2.imwrite(str(out_path), annotated)

            decision = "polyp (box)" if (conf >= threshold and bbox is not None) else "uncertain (circle)"
            results.append((img_path.name, conf, decision))
            print(f"  {img_path.name:<40} conf={conf*100:5.1f}%  -> {decision}")

    print(f"\nSaved {len(results)} annotated image(s) to {out_dir}")
    n_uncertain = sum(1 for _, _, d in results if "uncertain" in d)
    print(f"  {len(results) - n_uncertain} confident (box) | {n_uncertain} uncertain (circle)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--input_dir", type=str, default=None,
                         help="Folder of images to run on. Default: the held-out test set from config.yaml")
    parser.add_argument("--threshold", type=float, default=0.85,
                         help="Confidence threshold for drawing a box vs an uncertain-flag circle")
    parser.add_argument("--n", type=int, default=None, help="Optional cap on number of images to process")
    args = parser.parse_args()
    main(args.config, args.input_dir, args.threshold, args.n)