"""
inference_demo.py
Loads the best trained checkpoint and runs it on a folder of images, then
annotates each image based on the model's confidence there's a polyp:

  - confidence < 85%  -> draws a red circle in the top-left corner (flagged
                          "uncertain / needs manual review" - nothing drawn
                          on the tissue itself, since the mask isn't trusted
                          enough to localize anything at that confidence level)
  - confidence >= 85% -> marks the predicted polyp region (largest connected
                          component of the thresholded probability mask) with
                          a light, configurable-opacity overlay plus a thin
                          boundary outline - tuned to stay out of the way of
                          someone actually working from the image, not just
                          reviewing it after the fact

"Confidence" is a single per-image number derived from the per-pixel sigmoid
probability map: the mean of the top TOP_PIXEL_FRACTION most confident pixels,
not just the single max pixel - one bright outlier pixel would otherwise
always read as ~100% confident even on a genuinely polyp-free frame.

Usage:
    python inference_demo.py --config config.yaml --input_dir "path/to/images"
    python inference_demo.py --config config.yaml                     # defaults to held-out test set
    python inference_demo.py --config config.yaml --alpha 0.15        # lighter fill
    python inference_demo.py --config config.yaml --outline_only      # no fill at all, just a boundary line
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

RED = (0, 0, 255)      # BGR for cv2
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


def largest_component_mask(binary_mask):
    """Returns a binary mask containing ONLY the largest connected component
    of `binary_mask` (same shape, 0/1), or None if the mask is empty. Used
    instead of a bounding box so the overlay follows the polyp's actual
    predicted shape rather than a rectangle around it."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:  # only the background label (0) found
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    return (labels == largest_label).astype(np.uint8)


def annotate(image_bgr, confidence, threshold, component_mask=None, alpha=0.25, outline_only=False):
    """Draws either a red 'uncertain' circle (top-left) or a marked polyp
    region, plus a confidence label, on a copy of the original-resolution
    BGR image. alpha=0 (or outline_only=True) leaves the tissue itself fully
    visible and marks only the boundary."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    label = f"{confidence*100:.1f}%"

    confident_and_localized = (confidence >= threshold) and (component_mask is not None)

    if confident_and_localized:
        if not outline_only and alpha > 0:
            mask_bool = component_mask.astype(bool)
            colored = out.copy()
            colored[mask_bool] = GREEN
            out = cv2.addWeighted(colored, alpha, out, 1 - alpha, 0)

        # boundary outline is always drawn, however light the fill - this is
        # what stays legible even at alpha close to 0
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, GREEN, 2)

        ys, xs = np.where(component_mask.astype(bool))
        text_x, text_y = int(xs.min()), max(15, int(ys.min()) - 8)
        cv2.putText(out, f"polyp {label}", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
    else:
        # low confidence, OR confidence >= threshold but the thresholded mask
        # came up empty (can happen right at the boundary) - flag rather than
        # mark nothing
        radius = max(10, int(min(h, w) * 0.04))
        center = (radius + 10, radius + 10)
        cv2.circle(out, center, radius, RED, thickness=-1)
        cv2.putText(out, f"uncertain {label}", (center[0] + radius + 8, center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
    return out


def main(cfg_path, input_dir, threshold, n_max, alpha, outline_only):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.get("model_variant", "modified"), num_classes=cfg["num_classes"],
                         pretrained=False, segformer_size=cfg.get("segformer_size", "b0")).to(device)
    ckpt_path = Path(cfg["checkpoint_dir"]) / "best_model.pt"
    # weights_only=False: this is our own checkpoint - PyTorch 2.6's stricter
    # default can otherwise reject it depending on how/what torch version saved it.
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    print(f"Loaded best checkpoint from {ckpt_path}")
    print(f"Overlay style: {'outline only, no fill' if outline_only else f'fill alpha={alpha}'}")

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
            component_mask = largest_component_mask(binary_mask) if conf >= threshold else None

            annotated = annotate(image_bgr, conf, threshold, component_mask,
                                  alpha=alpha, outline_only=outline_only)
            out_path = out_dir / img_path.name
            cv2.imwrite(str(out_path), annotated)

            decision = "polyp (marked)" if (conf >= threshold and component_mask is not None) else "uncertain (circle)"
            results.append((img_path.name, conf, decision))
            print(f"  {img_path.name:<40} conf={conf*100:5.1f}%  -> {decision}")

    print(f"\nSaved {len(results)} annotated image(s) to {out_dir}")
    n_uncertain = sum(1 for _, _, d in results if "uncertain" in d)
    print(f"  {len(results) - n_uncertain} confident (marked) | {n_uncertain} uncertain (circle)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--input_dir", type=str, default=None,
                         help="Folder of images to run on. Default: the held-out test set from config.yaml")
    parser.add_argument("--threshold", type=float, default=0.85,
                         help="Confidence threshold for marking the polyp vs an uncertain-flag circle")
    parser.add_argument("--n", type=int, default=None, help="Optional cap on number of images to process")
    parser.add_argument("--alpha", type=float, default=0.25,
                         help="Fill opacity for the polyp overlay, 0.0-1.0 (default 0.25 - light enough "
                              "to keep tissue detail visible underneath). Ignored if --outline_only is set.")
    parser.add_argument("--outline_only", action="store_true",
                         help="Draw only the boundary line around the predicted polyp, no fill at all - "
                              "leaves the tissue fully visible for someone actively working from the image.")
    args = parser.parse_args()
    main(args.config, args.input_dir, args.threshold, args.n, args.alpha, args.outline_only)