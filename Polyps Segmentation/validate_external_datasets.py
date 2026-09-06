"""Evaluate the trained segmentation model on datasets outside the training set.

The script scans each top-level directory under --datasets-root, excludes every
dataset referenced by config.yaml, and reports model-only and end-to-end timing.
Dice/IoU/pixel accuracy are reported when a raster ground-truth mask can be paired
with an image; image-only, classification, and detection datasets are timed only.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from model import build_model


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_DIR_NAMES = {"mask", "masks", "ground truth", "ground_truth", "gt", "annotation", "annotations", "ann"}
AUXILIARY_SUFFIXES = ("_mask", "_gt", "_label", "_depth", "_normals", "_normal", "_flow", "_occlusion")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolved(path_value, base=ROOT):
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def is_mask_or_auxiliary(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    stem = path.stem.lower()
    return (
        bool(parts & MASK_DIR_NAMES)
        or stem.startswith(("mask_", "gt_"))
        or stem.endswith(AUXILIARY_SUFFIXES)
    )


def discover_images(dataset_dir: Path):
    return sorted(
        path for path in dataset_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not is_mask_or_auxiliary(path.relative_to(dataset_dir))
    )


def mask_candidates(image_path: Path):
    """Generate common same-folder and parallel-folder raster-mask paths."""
    stems = (image_path.stem, f"mask_{image_path.stem}", f"gt_{image_path.stem}", f"p{image_path.stem}")
    extensions = (image_path.suffix, ".png", ".jpg", ".tif", ".tiff", ".bmp")
    candidates = []

    for stem in stems:
        for extension in extensions:
            candidates.append(image_path.with_name(stem + extension))

    replacements = {
        "images": ("masks", "mask", "Ground Truth", "ground_truth", "annotations", "ann"),
        "image": ("masks", "mask", "Ground Truth", "ground_truth", "annotations", "ann"),
        "img": ("ann", "annotations", "masks", "mask"),
        "original": ("Ground Truth", "masks", "mask"),
        "frames": ("annotations", "masks", "mask"),
    }
    parts = list(image_path.parts)
    for index, part in enumerate(parts[:-1]):
        for replacement in replacements.get(part.lower(), ()):
            replaced = parts.copy()
            replaced[index] = replacement
            parent = Path(*replaced[:-1])
            for stem in stems:
                for extension in extensions:
                    candidates.append(parent / (stem + extension))
    return candidates


def find_mask(image_path: Path):
    image_resolved = image_path.resolve()
    for candidate in mask_candidates(image_path):
        if candidate.exists() and candidate.resolve() != image_resolved and is_mask_or_auxiliary(candidate):
            return candidate
    return None


def read_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.dtype != np.uint8:
        maximum = float(image.max())
        image = np.zeros_like(image, dtype=np.uint8) if maximum == 0 else np.clip(image / maximum * 255, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def preprocess(image_bgr, image_size):
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = (image.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


def load_mask(path: Path, image_size: int):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)


def segmentation_scores(probabilities, targets, threshold):
    predictions = probabilities >= threshold
    targets = targets >= 0.5
    dims = (1, 2, 3)
    intersection = (predictions & targets).sum(dim=dims).float()
    pred_sum = predictions.sum(dim=dims).float()
    target_sum = targets.sum(dim=dims).float()
    union = pred_sum + target_sum - intersection
    dice = torch.where(pred_sum + target_sum == 0, 1.0, 2 * intersection / (pred_sum + target_sum))
    iou = torch.where(union == 0, 1.0, intersection / union)
    accuracy = (predictions == targets).flatten(1).float().mean(dim=1)
    return dice.cpu().tolist(), iou.cpu().tolist(), accuracy.cpu().tolist()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_dataset(model, paths, image_size, batch_size, threshold, device, warmup_batches):
    totals = {"images": 0, "paired_masks": 0, "unreadable": 0, "forward_seconds": 0.0, "end_to_end_seconds": 0.0}
    dice_values, iou_values, accuracy_values = [], [], []
    if warmup_batches:
        sample = torch.zeros((1, 3, image_size, image_size), device=device)
        with torch.inference_mode():
            for _ in range(warmup_batches):
                model(sample)
        synchronize(device)

    for start in range(0, len(paths), batch_size):
        batch_started = time.perf_counter()
        valid_paths, tensors, masks = [], [], []
        for path in paths[start:start + batch_size]:
            image = read_image(path)
            if image is None:
                totals["unreadable"] += 1
                continue
            valid_paths.append(path)
            tensors.append(preprocess(image, image_size))
            mask_path = find_mask(path)
            masks.append(load_mask(mask_path, image_size) if mask_path else None)

        if not tensors:
            continue
        inputs = torch.stack(tensors).to(device)
        synchronize(device)
        forward_started = time.perf_counter()
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(inputs))
        synchronize(device)
        totals["forward_seconds"] += time.perf_counter() - forward_started
        totals["end_to_end_seconds"] += time.perf_counter() - batch_started
        totals["images"] += len(valid_paths)

        paired_indices = [index for index, mask in enumerate(masks) if mask is not None]
        if paired_indices:
            targets = torch.stack([masks[index] for index in paired_indices]).to(device)
            selected = probabilities[paired_indices]
            dice, iou, accuracy = segmentation_scores(selected, targets, threshold)
            dice_values.extend(dice)
            iou_values.extend(iou)
            accuracy_values.extend(accuracy)
            totals["paired_masks"] += len(paired_indices)

    count = totals["images"]
    return {
        **totals,
        "avg_inference_ms": totals["forward_seconds"] * 1000 / count if count else None,
        "avg_end_to_end_ms": totals["end_to_end_seconds"] * 1000 / count if count else None,
        "mean_dice": float(np.mean(dice_values)) if dice_values else None,
        "mean_iou": float(np.mean(iou_values)) if iou_values else None,
        "pixel_accuracy": float(np.mean(accuracy_values)) if accuracy_values else None,
    }


def write_reports(results, output_dir: Path, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"metadata": metadata, "datasets": results}
    (output_dir / "external_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fields = ["dataset", "images", "paired_masks", "unreadable", "avg_inference_ms", "avg_end_to_end_ms", "mean_dice", "mean_iou", "pixel_accuracy"]
    with (output_dir / "external_validation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, values in results.items():
            writer.writerow({"dataset": name, **{field: values.get(field) for field in fields[1:]}})


def balanced_sample(dataset_paths, total_images, seed):
    """Select up to total_images, distributing samples across datasets."""
    rng = random.Random(seed)
    available = {name: list(paths) for name, paths in dataset_paths.items() if paths}
    quotas = {name: 0 for name in available}
    remaining = min(total_images, sum(len(paths) for paths in available.values()))

    # Round-robin allocation keeps the sample as balanced as dataset sizes allow.
    while remaining:
        progressed = False
        for name, paths in available.items():
            if quotas[name] < len(paths) and remaining:
                quotas[name] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    selected = {}
    for name, paths in available.items():
        count = quotas[name]
        if count:
            selected[name] = sorted(rng.sample(paths, count))
    return selected


def main(args):
    with args.config.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = args.checkpoint or resolved(cfg["checkpoint_dir"])
    if checkpoint.is_dir():
        checkpoint = checkpoint / "best_model.pt"

    model = build_model(cfg.get("model_variant", "modified"), cfg.get("num_classes", 1), pretrained=False, segformer_size=cfg.get("segformer_size", "b0")).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    excluded_names = {name.casefold() for name in cfg["datasets"]}
    excluded_roots = set()
    datasets_root_resolved = args.datasets_root.resolve()
    for item in cfg["datasets"].values():
        image_dir = resolved(item["images_dir"])
        try:
            top_level_name = image_dir.relative_to(datasets_root_resolved).parts[0]
            excluded_roots.add((datasets_root_resolved / top_level_name).resolve())
        except (ValueError, IndexError):
            pass
    requested = {name.casefold() for name in args.dataset} if args.dataset else None
    dataset_dirs = [path for path in args.datasets_root.iterdir() if path.is_dir()]
    dataset_dirs = [path for path in dataset_dirs if path.resolve() not in excluded_roots and path.name.casefold() not in excluded_names]
    if requested:
        dataset_dirs = [path for path in dataset_dirs if path.name.casefold() in requested]

    dataset_paths = {}
    for dataset_dir in sorted(dataset_dirs):
        paths = discover_images(dataset_dir)
        if args.max_images:
            paths = paths[:args.max_images]
        if paths:
            dataset_paths[dataset_dir.name] = paths
        else:
            print(f"[SKIP] {dataset_dir.name}: no eligible images")
    if args.total_images:
        dataset_paths = balanced_sample(dataset_paths, args.total_images, args.seed)

    results = {}
    all_forward_seconds = 0.0
    all_end_to_end_seconds = 0.0
    all_images = 0
    for dataset_name, paths in dataset_paths.items():
        print(f"[RUN ] {dataset_name}: {len(paths)} image(s)")
        result = evaluate_dataset(model, paths, int(cfg["image_size"]), args.batch_size, args.threshold, device, args.warmup_batches)
        results[dataset_name] = result
        all_forward_seconds += result["forward_seconds"]
        all_end_to_end_seconds += result["end_to_end_seconds"]
        all_images += result["images"]
        print(f"       inference={result['avg_inference_ms']:.2f} ms/image, masks={result['paired_masks']}, Dice={result['mean_dice']}")

    total_paired_masks = sum(value["paired_masks"] for value in results.values())
    def weighted_metric(name):
        if not total_paired_masks:
            return None
        return sum(value[name] * value["paired_masks"] for value in results.values() if value[name] is not None) / total_paired_masks

    results["OVERALL"] = {
        "images": all_images,
        "paired_masks": total_paired_masks,
        "unreadable": sum(value["unreadable"] for value in results.values()),
        "avg_inference_ms": all_forward_seconds * 1000 / all_images if all_images else None,
        "avg_end_to_end_ms": all_end_to_end_seconds * 1000 / all_images if all_images else None,
        "mean_dice": weighted_metric("mean_dice"),
        "mean_iou": weighted_metric("mean_iou"),
        "pixel_accuracy": weighted_metric("pixel_accuracy"),
    }
    metadata = {"checkpoint": str(checkpoint), "device": str(device), "image_size": cfg["image_size"], "threshold": args.threshold, "batch_size": args.batch_size, "seed": args.seed, "excluded_training_datasets": sorted(cfg["datasets"])}
    write_reports(results, args.output_dir, metadata)
    print(f"\nSaved reports to {args.output_dir}")
    print(json.dumps(results["OVERALL"], indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=Path(r"C:\Users\Omen Max\Datasets"))
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "external_validation")
    parser.add_argument("--dataset", action="append", default=[], help="Only run this top-level dataset name; repeat as needed")
    parser.add_argument("--max-images", type=int, help="Limit images per dataset for a quick benchmark")
    parser.add_argument("--total-images", type=int, help="Balanced, reproducible sample across all eligible datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used by --total-images")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
