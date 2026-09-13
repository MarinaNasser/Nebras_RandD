"""
dataset.py
Loads image/mask pairs from any datasets listed in config.yaml, and builds
splits for several training protocols:

  - build_cross_dataset_splits(cfg): leave-one-out protocol - pool all
    datasets except `held_out_dataset` for training, hold that one out
    entirely for testing. Used by train.py / run_loo_experiment.py.

  - build_pooled_splits(cfg): pools ALL datasets, split per-dataset into
    train/val/test. Used by train_pooled.py.

  - list_negative_only_pairs(...): images with NO polyp (e.g. frames where
    only the endoscope/instrument is visible), paired with a synthetic
    all-zero mask generated on the fly - no mask files needed on disk. Used
    by finetune_scope_negatives.py for hard-negative fine-tuning.
"""
import os
from pathlib import Path
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import cv2

if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# Sentinel mask_path value meaning "this image has no polyp - use an
# all-zero mask matching the image's own dimensions, generated on the fly".
# Lets negative-only images (e.g. scope-only frames) flow through the exact
# same PolypDataset/TaggedPolypDataset code path as real (image, mask) pairs,
# with no mask files needing to exist on disk.
ZERO_MASK_SENTINEL = "__ZERO_MASK__"


def match_mask_for_image(img_path: Path, masks_dir: Path, mask_ext: str) -> Path:
    """Maps an image file to its corresponding mask file by filename stem."""
    candidate = masks_dir / f"{img_path.stem}{mask_ext}"
    if candidate.exists():
        return candidate
    for prefix in ("mask_", "p", "gt_"):
        alt = masks_dir / f"{prefix}{img_path.stem}{mask_ext}"
        if alt.exists():
            return alt
    raise FileNotFoundError(
        f"No mask found for image '{img_path.name}' in '{masks_dir}'. "
        f"Check dataset paths / naming in config.yaml."
    )


def list_pairs(images_dir: str, masks_dir: str, img_ext: str, mask_ext: str, dataset_name: str = None):
    """Returns a list of (image_path, mask_path) or (image_path, mask_path, dataset_name) tuples."""
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir does not exist: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks_dir does not exist: {masks_dir}")

    img_paths = sorted([p for p in images_dir.glob(f"*{img_ext}")])
    if not img_paths:
        raise FileNotFoundError(
            f"No '{img_ext}' images found in {images_dir}. Check img_ext in config.yaml."
        )

    pairs = []
    for img_path in img_paths:
        mask_path = match_mask_for_image(img_path, masks_dir, mask_ext)
        if dataset_name is not None:
            pairs.append((str(img_path), str(mask_path), dataset_name))
        else:
            pairs.append((str(img_path), str(mask_path)))
    return pairs
def list_scope_negative_pairs(images_dir, img_ext):
    p = Path(images_dir)
    if not p.exists():
        raise FileNotFoundError(f"Scope images directory does not exist: {images_dir}")

    # Clean extension formatting (e.g., handles "bmp", ".bmp", ".BMP")
    clean_ext = img_ext.strip().lstrip(".")
    
    # Use rglob to find images even if nested inside subfolders
    patterns = [f"*.{clean_ext.lower()}", f"*.{clean_ext.upper()}"]
    
    image_paths = []
    seen = set()
    for pat in patterns:
        for f in p.rglob(pat):
            if f.is_file() and f.resolve() not in seen:
                image_paths.append(str(f))
                seen.add(f.resolve())

    image_paths = sorted(image_paths)

    if not image_paths:
        # Check if the folder contains any files at all to give a clear diagnostic
        all_files = list(p.rglob("*.*"))
        sample_exts = set(f.suffix for f in all_files[:20])
        raise FileNotFoundError(
            f"Found 0 '{img_ext}' files in '{images_dir}'. "
            f"Extensions detected in folder: {sample_exts if sample_exts else 'No files found'}."
        )

    return [(img_p, ZERO_MASK_SENTINEL, "scope_negative") for img_p in image_paths]

def get_train_transforms(image_size: int):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def _read_pair_raw(img_path: str, mask_path: str):
    """Reads one (image, mask) pair. mask_path == ZERO_MASK_SENTINEL generates
    an all-zero mask matching the image's dimensions instead of reading from
    disk. Returns None if the image (or a real mask file) is unreadable."""
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if mask_path == ZERO_MASK_SENTINEL:
        mask = np.zeros(image.shape[:2], dtype=np.float32)
    else:
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            return None
        mask = (mask_img > 127).astype(np.float32)
    return image, mask


class PolypDataset(Dataset):
    """pairs: list of (image_path, mask_path). mask_path may be
    ZERO_MASK_SENTINEL for a negative-only (no-polyp) image."""
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        max_retries = 5
        for attempt in range(max_retries):
            img_path, mask_path = self.pairs[idx]
            result = _read_pair_raw(img_path, mask_path)
            if result is not None:
                image, mask = result
                break
            print(f"[WARN] Skipping unreadable pair (attempt {attempt+1}/{max_retries}): "
                  f"image={img_path} mask={mask_path}")
            idx = np.random.randint(0, len(self.pairs))
        else:
            raise RuntimeError(
                f"Failed to find a readable image/mask pair after {max_retries} retries. "
                f"Too many corrupt files - run validate_datasets.py."
            )

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0) if mask.ndim == 2 else mask
        return image, mask.float()


class TaggedPolypDataset(Dataset):
    """Like PolypDataset, but pairs are (image_path, mask_path, dataset_name)
    triples and __getitem__ also returns dataset_name."""
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        max_retries = 5
        for attempt in range(max_retries):
            img_path, mask_path, dataset_name = self.pairs[idx]
            result = _read_pair_raw(img_path, mask_path)
            if result is not None:
                image, mask = result
                break
            print(f"[WARN] Skipping unreadable pair (attempt {attempt+1}/{max_retries}): "
                  f"image={img_path} mask={mask_path}")
            idx = np.random.randint(0, len(self.pairs))
        else:
            raise RuntimeError(
                f"Failed to find a readable image/mask pair after {max_retries} retries. "
                f"Too many corrupt files - run validate_datasets.py."
            )

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0) if mask.ndim == 2 else mask
        return image, mask.float(), dataset_name


def build_cross_dataset_splits(cfg):
    """
    Builds:
      - train_pairs: pooled pairs from every dataset EXCEPT the held-out one, minus a
        small internal validation slice (for early stopping / model selection)
      - internal_val_pairs: that slice, drawn from the SAME training datasets
      - test_pairs: all pairs from the held-out dataset (never touched during training)
    """
    held_out = cfg["held_out_dataset"]
    ds_cfg = cfg["datasets"]
    assert held_out in ds_cfg, f"held_out_dataset '{held_out}' not found in config datasets"

    train_names = [name for name in ds_cfg if name != held_out]
    assert len(train_names) >= 2, (
        "Need at least 2 datasets total (1+ for training, 1 held out for validation). "
        f"Got {len(ds_cfg)} dataset(s) configured."
    )

    rng = np.random.RandomState(cfg["seed"])

    pooled_train_pairs = []
    for name in train_names:
        d = ds_cfg[name]
        pairs = list_pairs(d["images_dir"], d["masks_dir"], d["img_ext"], d["mask_ext"])
        print(f"[{name}] {len(pairs)} image/mask pairs found (training pool)")
        pooled_train_pairs.extend(pairs)

    rng.shuffle(pooled_train_pairs)
    n_internal_val = int(len(pooled_train_pairs) * cfg["internal_val_fraction"])
    internal_val_pairs = pooled_train_pairs[:n_internal_val]
    train_pairs = pooled_train_pairs[n_internal_val:]

    d = ds_cfg[held_out]
    test_pairs = list_pairs(d["images_dir"], d["masks_dir"], d["img_ext"], d["mask_ext"])
    print(f"[{held_out}] {len(test_pairs)} image/mask pairs found (held-out TEST set)")

    print(f"\nFinal split -> train: {len(train_pairs)} | internal val: {len(internal_val_pairs)} "
          f"| held-out test ({held_out}): {len(test_pairs)}\n")

    return train_pairs, internal_val_pairs, test_pairs


def build_pooled_splits(cfg):
    """
    Pools ALL datasets, split per-dataset into train/val/test by fractions
    in cfg["pooled_split"] (train_fraction/val_fraction/test_fraction, sum
    to 1.0). Returns (image_path, mask_path, dataset_name) triples for
    train_pairs, val_pairs, test_pairs.
    """
    split_cfg = cfg["pooled_split"]
    train_frac = float(split_cfg["train_fraction"])
    val_frac = float(split_cfg["val_fraction"])
    test_frac = float(split_cfg["test_fraction"])
    total = train_frac + val_frac + test_frac
    assert abs(total - 1.0) < 1e-6, f"pooled_split fractions must sum to 1.0, got {total}"

    rng = np.random.RandomState(cfg["seed"])

    train_pairs, val_pairs, test_pairs = [], [], []
    print("Pooled train/val/test split (per-dataset stratified):")
    for name, d in cfg["datasets"].items():
        pairs = list_pairs(d["images_dir"], d["masks_dir"], d["img_ext"], d["mask_ext"])
        pairs = [(img, mask, name) for img, mask in pairs]
        rng.shuffle(pairs)

        n = len(pairs)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        d_train = pairs[:n_train]
        d_val = pairs[n_train:n_train + n_val]
        d_test = pairs[n_train + n_val:]

        train_pairs.extend(d_train)
        val_pairs.extend(d_val)
        test_pairs.extend(d_test)

        print(f"  [{name}] total={n} -> train={len(d_train)} val={len(d_val)} test={len(d_test)}")

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    rng.shuffle(test_pairs)

    print(f"\nOverall -> train: {len(train_pairs)} | val: {len(val_pairs)} | test: {len(test_pairs)}\n")
    return train_pairs, val_pairs, test_pairs


def build_stratified_splits(cfg):
    """
    Splits EACH dataset independently according to train/val/test fractions,
    then pools them together.
    """
    train_frac = cfg.get("train_fraction", 0.70)
    val_frac = cfg.get("val_fraction", 0.10)
    rng = np.random.RandomState(cfg["seed"])

    train_pairs, val_pairs, test_pairs = [], [], []

    print("\n--- Dataset Split Breakdown ---")
    for name, d in cfg["datasets"].items():
        pairs = list_pairs(d["images_dir"], d["masks_dir"], d["img_ext"], d["mask_ext"], dataset_name=name)
        rng.shuffle(pairs)

        n_total = len(pairs)
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)

        ds_train = pairs[:n_train]
        ds_val = pairs[n_train:n_train + n_val]
        ds_test = pairs[n_train + n_val:]

        train_pairs.extend(ds_train)
        val_pairs.extend(ds_val)
        test_pairs.extend(ds_test)

        print(f"[{name:<18}] Total: {n_total:<5} | Train: {len(ds_train):<5} | Val: {len(ds_val):<5} | Test: {len(ds_test):<5}")

    print(f"\nPooled -> Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}\n")
    return train_pairs, val_pairs, test_pairs