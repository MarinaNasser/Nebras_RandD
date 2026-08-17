"""
dataset.py
Loads REAL-Colon video frames + bounding-box polyp annotations, and builds a
video-level (patient-level) train/val/test split, stratified by site, so no
frames from the same video ever appear in more than one split.

REAL-Colon layout expected under cfg["root_dir"]:
  SSS-VVV_frames/         one folder per video, containing frame jpgs
  SSS-VVV_annotations/    matching folder, one xml per frame (PASCAL-VOC-style,
                           possibly zero <object> elements for a no-polyp frame)
"""
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def parse_annotation_xml(xml_path: Path):
    """Returns a list of [xmin, ymin, xmax, ymax] boxes (pixel coords) for one frame.
    A frame with zero <object> elements is a valid negative (non-polyp) frame."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        if xmax > xmin and ymax > ymin:  # guard against degenerate boxes
            boxes.append([xmin, ymin, xmax, ymax])
    return boxes


def discover_videos(root_dir: str, frames_suffix="_frames", annotations_suffix="_annotations"):
    """
    Scans root_dir for REAL-Colon's per-video folder pairs:
      SSS-VVV_frames/        (jpgs)
      SSS-VVV_annotations/   (one xml per frame)
    Returns {video_id: {"frames_dir": Path, "annotations_dir": Path, "site_id": str}}
    site_id is the "SSS" prefix of the video id (e.g. "001-014" -> site "001"),
    which REAL-Colon uses to distinguish its contributing centers.
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir does not exist: {root_dir}")

    videos = {}
    for frames_dir in sorted(root_dir.glob(f"*{frames_suffix}")):
        video_id = frames_dir.name[: -len(frames_suffix)]
        annotations_dir = root_dir / f"{video_id}{annotations_suffix}"
        if not annotations_dir.exists():
            raise FileNotFoundError(
                f"Found frames dir '{frames_dir.name}' but no matching annotations dir "
                f"'{annotations_dir.name}'. Check your REAL-Colon download / root_dir path."
            )
        site_id = video_id.split("-")[0]
        videos[video_id] = {"frames_dir": frames_dir, "annotations_dir": annotations_dir, "site_id": site_id}

    if not videos:
        raise FileNotFoundError(
            f"No '*{frames_suffix}' folders found under {root_dir}. "
            f"Check config.yaml's root_dir points at the REAL-Colon download root."
        )
    return videos


def split_videos_by_site(videos: dict, train_frac: float, val_frac: float, test_frac: float, seed: int):
    """
    Splits video IDs into train/val/test, stratified by site, so every split
    gets a proportional mix of contributing centers rather than, say, the test
    split accidentally landing entirely on one hospital's videos. Critically,
    every frame of a given video stays in exactly one split - never splitting
    within a video - since consecutive frames of the same polyp are highly
    correlated and would otherwise leak information between splits.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "split fractions must sum to 1.0"
    rng = random.Random(seed)

    by_site = {}
    for vid, info in videos.items():
        by_site.setdefault(info["site_id"], []).append(vid)

    train_ids, val_ids, test_ids = [], [], []
    for site_id, vids in by_site.items():
        vids = sorted(vids)
        rng.shuffle(vids)
        n = len(vids)
        n_train = max(1, round(n * train_frac)) if n >= 3 else n
        n_train = min(n_train, n)
        n_val = max(0, min(round(n * val_frac), n - n_train))
        train_ids += vids[:n_train]
        val_ids += vids[n_train:n_train + n_val]
        test_ids += vids[n_train + n_val:]

    return train_ids, val_ids, test_ids


def list_frame_pairs(video_ids, videos: dict, img_ext=".jpg",
                      frame_stride=1, negative_frame_sample_ratio=1.0, seed=0):
    """
    For each video in video_ids, walks its frames_dir, matches each frame to its
    annotation xml by filename stem, and returns a list of
    (image_path, [[xmin,ymin,xmax,ymax], ...]) tuples. Empty box list = negative frame.

    frame_stride: keep every Nth frame. Video frames are highly redundant
        (consecutive frames barely differ), and REAL-Colon is ~2.7M frames
        across 60 videos - striding is necessary for a practical dataset size.
    negative_frame_sample_ratio: fraction of *negative* (no-polyp) frames to
        keep, after striding, applied independently per frame. Positive frames
        (>=1 polyp box) are always kept in full, so this only controls
        foreground/background imbalance, never throws away positive supervision.
    """
    rng = random.Random(seed)
    pairs = []
    for vid in video_ids:
        info = videos[vid]
        img_paths = sorted(info["frames_dir"].glob(f"*{img_ext}"))[::frame_stride]
        for img_path in img_paths:
            xml_path = info["annotations_dir"] / f"{img_path.stem}.xml"
            if not xml_path.exists():
                continue  # frame with no annotation file is skipped, not treated as negative
            boxes = parse_annotation_xml(xml_path)
            if not boxes and rng.random() > negative_frame_sample_ratio:
                continue
            pairs.append((str(img_path), boxes))
    return pairs


def get_train_transforms(image_size: int):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.3),
    )


def get_val_transforms(image_size: int):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.0),
    )


class RealColonDetectionDataset(Dataset):
    """
    pairs: list of (image_path, [[xmin,ymin,xmax,ymax], ...])
    Returns (image_tensor, target_dict) with "boxes" (FloatTensor[N,4]) and
    "labels" (Int64Tensor[N], all 1 = "polyp") - the format torchvision's
    detection models expect.
    """
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, boxes = self.pairs[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = boxes if boxes else []
        labels = [1] * len(boxes)  # single foreground class: "polyp"

        if self.transform:
            augmented = self.transform(image=image, bboxes=boxes, labels=labels)
            image = augmented["image"]
            boxes = augmented["bboxes"]
            labels = augmented["labels"]

        if len(boxes) > 0:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t, "image_id": torch.tensor([idx])}
        return image, target


def detection_collate_fn(batch):
    """torchvision detection models take a list of images + list of target dicts,
    not a stacked batch tensor (images/box-counts vary per sample)."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_realcolon_splits(cfg):
    """
    Discovers all REAL-Colon videos under cfg["root_dir"], splits them
    video-/site-stratified into train/val/test per cfg["split"], then lists
    frame/annotation pairs for each split (with striding + negative-frame
    subsampling per cfg["sampling"]).
    """
    videos = discover_videos(cfg["root_dir"])
    n_sites = len(set(v["site_id"] for v in videos.values()))
    print(f"Discovered {len(videos)} REAL-Colon videos across {n_sites} sites")

    train_ids, val_ids, test_ids = split_videos_by_site(
        videos,
        train_frac=cfg["split"]["train_frac"],
        val_frac=cfg["split"]["val_frac"],
        test_frac=cfg["split"]["test_frac"],
        seed=cfg["seed"],
    )
    print(f"Video split -> train: {len(train_ids)} videos | val: {len(val_ids)} videos | "
          f"test: {len(test_ids)} videos (a video never appears in more than one split)")

    sampling = cfg["sampling"]
    train_pairs = list_frame_pairs(
        train_ids, videos, frame_stride=sampling["frame_stride"],
        negative_frame_sample_ratio=sampling["negative_frame_sample_ratio"], seed=cfg["seed"])
    val_pairs = list_frame_pairs(
        val_ids, videos, frame_stride=sampling["frame_stride"],
        negative_frame_sample_ratio=sampling["negative_frame_sample_ratio"], seed=cfg["seed"])
    # test set: keep every negative frame (after striding) so held-out metrics
    # reflect the real positive/negative frame ratio, not an artificially
    # boosted one - subsampling negatives is a training-time convenience only.
    test_pairs = list_frame_pairs(
        test_ids, videos, frame_stride=sampling["frame_stride"],
        negative_frame_sample_ratio=1.0, seed=cfg["seed"])

    def n_pos(pairs):
        return sum(1 for _, boxes in pairs if boxes)

    print(f"Train: {len(train_pairs)} frames ({n_pos(train_pairs)} with >=1 polyp)")
    print(f"Val:   {len(val_pairs)} frames ({n_pos(val_pairs)} with >=1 polyp)")
    print(f"Test:  {len(test_pairs)} frames ({n_pos(test_pairs)} with >=1 polyp)")

    return train_pairs, val_pairs, test_pairs
