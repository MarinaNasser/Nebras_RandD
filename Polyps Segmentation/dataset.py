"""
dataset.py
Loads image/mask pairs from CVC-ClinicDB, CVC-ColonDB, and ETIS-LaribPolypDB,
and builds the cross-dataset train/val split (train on 2, validate on the held-out 3rd).
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


def match_mask_for_image(img_path: Path, masks_dir: Path, mask_ext: str) -> Path:
    candidate = masks_dir / f"{img_path.stem}{mask_ext}"
    if candidate.exists():
        return candidate
    for prefix in ("mask_", "p", "gt_"):
        alt = masks_dir / f"{prefix}{img_path.stem}{mask_ext}"
        if alt.exists():
            return alt
    raise FileNotFoundError(f"No mask found for image '{img_path.name}' in '{masks_dir}'.")


def list_pairs(images_dir: str, masks_dir: str, img_ext: str, mask_ext: str, dataset_name: str):
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    img_paths = sorted([p for p in images_dir.glob(f"*{img_ext}")])
    if not img_paths:
        raise FileNotFoundError(f"No '{img_ext}' images found in {images_dir}.")

    pairs = []
    for img_path in img_paths:
        mask_path = match_mask_for_image(img_path, masks_dir, mask_ext)
        # Store (image_path, mask_path, dataset_name)
        pairs.append((str(img_path), str(mask_path), dataset_name))
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
        # A.GaussNoise(p=0.2),
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
    def __init__(self, pairs, transform=None):
        self.pairs = pairs  # list of (img_path, mask_path, dataset_name)
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path, ds_name = self.pairs[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0) if mask.ndim == 2 else mask
        return image, mask.float(), ds_name


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