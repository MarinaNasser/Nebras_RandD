"""
inference_demo.py
Loads the best trained checkpoint and runs it on a folder of images, then
annotates each image based on the model's confidence there's a polyp:

  - confidence < 85%  -> draws a red circle in the top-left corner (flagged
                          "uncertain / needs manual review" - nothing drawn
                          on the tissue itself)
  - confidence >= 85% -> marks the predicted polyp region (largest connected
                          component of the thresholded probability mask) with
                          a light, configurable-opacity overlay plus a thin
                          boundary outline, and reports its SIZE:

SIZE ESTIMATION
---------------
Always computed, in pixels:
    - area_px:            pixel count of the predicted polyp region
    - equiv_diameter_px:  diameter of a circle with the same area
                           (sqrt(4*area/pi)) - a single "how big" number
                           that isn't distorted by odd/elongated shapes
                           the way bounding-box width/height can be
    - pct_of_frame:       area_px / (image width * height) * 100

Real-world size (mm) is IMPOSSIBLE to get from pixels alone here: these are
monocular endoscopy frames with no fixed camera-to-tissue distance, so pixel
size isn't a constant real-world size across frames or datasets. If you have
a calibration constant (e.g. derived from a known reference object visible in
frame - open biopsy forceps jaw width, snare diameter, etc. - for a specific
scope/dataset), pass --pixel_to_mm and mm-based measurements + a rough Paris-
style size bucket (diminutive/small/large) are added on top. Without it, only
the pixel-based numbers are reported - don't present those as clinical
measurements, they're relative/comparative only (e.g. "this polyp took up
more of the frame than that one"), not physical sizes.

Usage:
    python inference_demo.py --config config.yaml --input_dir "path/to/images"
    python inference_demo.py --config config.yaml                     # defaults to held-out test set
    python inference_demo.py --config config.yaml --alpha 0.15        # lighter fill
    python inference_demo.py --config config.yaml --outline_only      # no fill, just a boundary line
    python inference_demo.py --config config.yaml --pixel_to_mm 0.045 # if you have a calibration constant
"""
import argparse
import csv
import math
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
    of `binary_mask` (same shape, 0/1), or None if the mask is empty."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:  # only the background label (0) found
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    return (labels == largest_label).astype(np.uint8)


def size_category_mm(diameter_mm):
    """Rough Paris-classification-style size bucket. Only meaningful if
    diameter_mm came from a real pixel_to_mm calibration."""
    if diameter_mm < 5:
        return "diminutive (<5mm)"
    elif diameter_mm < 10:
        return "small (6-9mm)"
    else:
        return "large (>=10mm)"


def measure_size(component_mask, orig_h, orig_w, pixel_to_mm=None):
    """Returns a dict of size metrics for one predicted polyp region."""
    area_px = int(component_mask.sum())
    equiv_diameter_px = math.sqrt(4 * area_px / math.pi)
    pct_of_frame = 100.0 * area_px / (orig_h * orig_w)

    metrics = {
        "area_px": area_px,
        "equiv_diameter_px": round(equiv_diameter_px, 1),
        "pct_of_frame": round(pct_of_frame, 2),
    }
    if pixel_to_mm is not None:
        diameter_mm = equiv_diameter_px * pixel_to_mm
        area_mm2 = area_px * (pixel_to_mm ** 2)
        metrics["diameter_mm"] = round(diameter_mm, 1)
        metrics["area_mm2"] = round(area_mm2, 1)
        metrics["size_category"] = size_category_mm(diameter_mm)
    return metrics


def annotate(image_bgr, confidence, threshold, component_mask=None, alpha=0.25,
             outline_only=False, size_metrics=None):
    """Draws either a red 'uncertain' circle (top-left) or a marked polyp
    region with a size label, on a copy of the original-resolution BGR image."""
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

        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, GREEN, 2)

        if size_metrics.get("diameter_mm") is not None:
            size_label = f"~{size_metrics['diameter_mm']:.0f}mm"
        else:
            size_label = f"~{size_metrics['equiv_diameter_px']:.0f}px"

        ys, xs = np.where(component_mask.astype(bool))
        text_x, text_y = int(xs.min()), max(15, int(ys.min()) - 8)
        cv2.putText(out, f"polyp {label} {size_label}", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
    else:
        radius = max(10, int(min(h, w) * 0.04))
        center = (radius + 10, radius + 10)
        cv2.circle(out, center, radius, RED, thickness=-1)
        cv2.putText(out, f"uncertain {label}", (center[0] + radius + 8, center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
    return out


def main(cfg_path, input_dir, threshold, n_max, alpha, outline_only, pixel_to_mm):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.get("model_variant", "modified"), num_classes=cfg["num_classes"],
                         pretrained=False, segformer_size=cfg.get("segformer_size", "b0")).to(device)
    ckpt_path = Path(cfg["checkpoint_dir"]) / "best_model.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    print(f"Loaded best checkpoint from {ckpt_path}")
    print(f"Overlay style: {'outline only, no fill' if outline_only else f'fill alpha={alpha}'}")
    if pixel_to_mm is not None:
        print(f"Size calibration: {pixel_to_mm} mm/pixel -> mm-based size estimates enabled")
    else:
        print("No --pixel_to_mm given: size estimates will be pixel-based only (relative, not physical mm)")

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
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

            conf = image_confidence(probs)

            probs_full = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            binary_mask = (probs_full > 0.5).astype(np.uint8)
            component_mask = largest_component_mask(binary_mask) if conf >= threshold else None

            size_metrics = {}
            if component_mask is not None:
                size_metrics = measure_size(component_mask, orig_h, orig_w, pixel_to_mm)

            annotated = annotate(image_bgr, conf, threshold, component_mask,
                                  alpha=alpha, outline_only=outline_only, size_metrics=size_metrics)
            out_path = out_dir / img_path.name
            cv2.imwrite(str(out_path), annotated)

            decision = "polyp (marked)" if (conf >= threshold and component_mask is not None) else "uncertain (circle)"
            row = {"filename": img_path.name, "confidence": round(conf, 4), "decision": decision}
            row.update(size_metrics)
            results.append(row)

            size_str = ""
            if size_metrics:
                if "diameter_mm" in size_metrics:
                    size_str = f" | ~{size_metrics['diameter_mm']}mm ({size_metrics['size_category']})"
                else:
                    size_str = f" | ~{size_metrics['equiv_diameter_px']}px, {size_metrics['pct_of_frame']}% of frame"
            print(f"  {img_path.name:<40} conf={conf*100:5.1f}%  -> {decision}{size_str}")

    csv_path = out_dir / "size_estimates.csv"
    fieldnames = ["filename", "confidence", "decision", "area_px", "equiv_diameter_px", "pct_of_frame"]
    if pixel_to_mm is not None:
        fieldnames += ["diameter_mm", "area_mm2", "size_category"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\nSaved {len(results)} annotated image(s) to {out_dir}")
    print(f"Saved per-image size estimates to {csv_path}")
    n_uncertain = sum(1 for r in results if "uncertain" in r["decision"])
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
                         help="Fill opacity for the polyp overlay, 0.0-1.0 (default 0.25). Ignored if --outline_only.")
    parser.add_argument("--outline_only", action="store_true",
                         help="Draw only the boundary line, no fill - leaves tissue fully visible.")
    parser.add_argument("--pixel_to_mm", type=float, default=None,
                         help="Calibration constant (millimeters per pixel) if known for this scope/dataset. "
                              "Without it, size is reported in pixels/percent-of-frame only (relative, not "
                              "a physical measurement).")
    args = parser.parse_args()
    main(args.config, args.input_dir, args.threshold, args.n, args.alpha, args.outline_only, args.pixel_to_mm)