"""
dataset.py
Loads image/mask pairs from CVC-ClinicDB, CVC-ColonDB, and ETIS-LaribPolypDB,
and builds the cross-dataset train/val split (train on 2, validate on the held-out 3rd).
"""
import os
from pathlib import Path
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import cv2

# Correct syntax for OpenCV logging
if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def match_mask_for_image(img_path: Path, masks_dir: Path, mask_ext: str) -> Path:
    """
    Maps an image file to its corresponding mask file by filename stem.
    Edit this function if your downloaded copies of the datasets use a
    different naming convention (e.g. masks prefixed with 'mask_', or a
    different numbering scheme between the image and ground-truth folders).
    """
    candidate = masks_dir / f"{img_path.stem}{mask_ext}"
    if candidate.exists():
        return candidate
    # fallback: try common prefixes some dataset mirrors use
    for prefix in ("mask_", "p", "gt_"):
        alt = masks_dir / f"{prefix}{img_path.stem}{mask_ext}"
        if alt.exists():
            return alt
    raise FileNotFoundError(
        f"No mask found for image '{img_path.name}' in '{masks_dir}'. "
        f"Check dataset paths / naming in config.yaml."
    )


def list_pairs(images_dir: str, masks_dir: str, img_ext: str, mask_ext: str):
    """Returns a list of (image_path, mask_path) tuples for one dataset."""
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
        pairs.append((str(img_path), str(mask_path)))
    return pairs


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


class PolypDataset(Dataset):
    """
    pairs: list of (image_path, mask_path)
    """
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        mask = (mask > 127).astype(np.float32)  # binarize

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0) if mask.ndim == 2 else mask  # (1, H, W)
        return image, mask.float()


def build_cross_dataset_splits(cfg):
    """
    Builds:
      - train_pairs: pooled pairs from every dataset EXCEPT the held-out one, minus a
        small internal validation slice (for early stopping / model selection)
      - internal_val_pairs: that slice, drawn from the SAME training datasets
      - test_pairs: all pairs from the held-out dataset (never touched during training)

    Any number of datasets >= 2 can be listed under cfg["datasets"] - all of them
    except `held_out_dataset` are pooled together for training. This is how you add
    an extra dataset (e.g. Kvasir-SEG) on top of the original 3: just add its entry
    to config.yaml's `datasets` block: it will automatically join the training pool
    for every held-out choice it isn't itself.
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
