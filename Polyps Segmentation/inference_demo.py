"""
inference_demo.py
Loads the trained checkpoint, runs inference on images, and draws green 
polyp contours directly over the full-color endoscopy frame.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from dataset import build_stratified_splits
from model import build_model

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

TOP_PIXEL_FRACTION = 0.01   # top 1% of pixels by predicted probability
MIN_TOP_PIXELS = 20         # floor for stable confidence estimation

# BGR Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def load_as_color_bgr(img_path):
    """
    Guarantees a 3-channel 8-bit BGR image even if the raw file is 
    a 16-bit, grayscale, or indexed/palette TIFF.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Normalize 16-bit TIFFs down to 8-bit
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)

    # Convert grayscale / single-channel to 3-channel BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def preprocess(image_bgr, image_size):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    normed = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(normed.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def image_confidence(prob_map):
    flat = prob_map.flatten()
    k = max(MIN_TOP_PIXELS, int(len(flat) * TOP_PIXEL_FRACTION))
    k = min(k, len(flat))
    top_k = np.partition(flat, -k)[-k:]
    return float(top_k.mean())


def annotate_contours(image_bgr, confidence, threshold, binary_mask=None):
    out = image_bgr.copy()
    h, w = out.shape[:2]
    label = f"{confidence * 100:.1f}%"

    has_detection = binary_mask is not None and binary_mask.sum() > 0
    confident_and_detected = (confidence >= threshold) and has_detection

    if confident_and_detected:
        # Find all external contours for detected polyps
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, GREEN, thickness=2)

        # Place label above the first detected bounding area
        all_pts = np.vstack(contours)
        x, y, _, _ = cv2.boundingRect(all_pts)
        text_y = max(22, y - 8)
        cv2.putText(out, f"Polyp ({label})", (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREEN, 2, cv2.LINE_AA)
    else:
        radius = max(10, int(min(h, w) * 0.035))
        center = (radius + 12, radius + 12)
        cv2.circle(out, center, radius, RED, thickness=-1)
        cv2.putText(out, f"Uncertain ({label})", (center[0] + radius + 8, center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)

    return out


def main(cfg_path, input_dir, threshold, n_max):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        cfg.get("model_variant", "modified"),
        num_classes=cfg["num_classes"],
        pretrained=False,
        segformer_size=cfg.get("segformer_size", "b0")
    ).to(device)

    ckpt_path = Path(cfg["checkpoint_dir"]) / "best_model.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    print(f"Loaded checkpoint from: {ckpt_path}")

    if input_dir:
        img_paths = sorted([p for p in Path(input_dir).iterdir()
                            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")])
        print(f"Running inference on {len(img_paths)} image(s) from: {input_dir}")
    else:
        _, _, test_pairs = build_stratified_splits(cfg)
        img_paths = [Path(p) for p, _, _ in test_pairs]
        print(f"No --input_dir given, defaulting to test set: {len(img_paths)} image(s)")

    if n_max:
        img_paths = img_paths[:n_max]

    out_dir = Path(cfg["output_dir"]) / "inference_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_size = cfg["image_size"]
    n_detected = 0

    with torch.no_grad():
        for img_path in img_paths:
            # Read explicitly ensuring 3-channel BGR color
            image_bgr = load_as_color_bgr(img_path)
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

            annotated = annotate_contours(image_bgr, conf, threshold, binary_mask)

            # Save as standard PNG/JPG to preserve full color display
            out_filename = img_path.stem + ".png"
            cv2.imwrite(str(out_dir / out_filename), annotated)

            if conf >= threshold and binary_mask.sum() > 0:
                n_detected += 1
                status = "Polyp Contoured"
            else:
                status = "Uncertain"

            print(f"  {img_path.name:<35} conf={conf*100:5.1f}% -> {status}")

    print(f"\nSaved {len(img_paths)} images to: {out_dir}")
    print(f"  {n_detected} detected with contours | {len(img_paths) - n_detected} uncertain/negative")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.input_dir, args.threshold, args.n)