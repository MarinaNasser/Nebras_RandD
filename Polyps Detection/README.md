# Polyp Detection on REAL-Colon (RetinaNet, ResNet-50 FPN)

Trains RetinaNet, pretrained on COCO, to detect polyps as bounding boxes in
REAL-Colon video frames. This is a **separate pipeline** from the SegNet
segmentation project - REAL-Colon's ground truth is bounding boxes, not
pixel masks, so it isn't compatible with the SegNet dataset/mask format and
needed its own model, dataset loader, and metrics.

## Why RetinaNet, and why this is a detection (not segmentation) task

REAL-Colon's 350k annotations are bounding boxes drawn by annotators around
each visible polyp per frame - there are no pixel-level masks, so Dice/IoU
segmentation metrics don't apply here. RetinaNet was chosen over a two-stage
detector (e.g. Faster R-CNN) because its focal loss down-weights the very
large number of easy background anchors during training, which suits
colonoscopy frames: most of each frame is textureless mucosa, and polyps are
frequently small or flat against that background.

## 1. Install

```bash
pip install torch torchvision albumentations opencv-python-headless pyyaml tqdm matplotlib numpy
```

## 2. Point the config at your data

REAL-Colon's download is organized as one folder pair per video:

```
REAL-Colon/
  001-001_frames/          jpgs
  001-001_annotations/     one xml per frame (PASCAL-VOC-style bndbox)
  001-002_frames/
  001-002_annotations/
  ...
```

Set `root_dir` in `config.yaml` to the folder containing all of these
`SSS-VVV_frames` / `SSS-VVV_annotations` pairs. Nothing else needs to change
to get started - the rest of `config.yaml` has reasonable defaults.

## 3. How the split works

Unlike the SegNet project (which held out an entire *dataset*), here all
train/val/test data comes from REAL-Colon itself, split at the **video**
level:

- Videos are grouped by contributing site (the `SSS` prefix of each video
  ID) and split `train_frac` / `val_frac` / `test_frac` **within each site**,
  so every split gets a proportional mix of centers.
- Every frame of a given video lands in exactly one split - never split
  frames from the same video across train/val/test, since consecutive
  frames of the same polyp are highly correlated and would otherwise leak
  information between splits and inflate your metrics.

## 4. Frame sampling

REAL-Colon is ~2.7M frames across 60 videos - far too many/too redundant to
train on directly (consecutive frames barely differ). Two knobs in
`config.yaml`'s `sampling` block control this:

- `frame_stride`: keep every Nth frame per video.
- `negative_frame_sample_ratio`: after striding, keep only this fraction of
  frames that have **no** polyp box. Frames with at least one polyp are
  always kept in full - this only trims background frames to control
  foreground/background imbalance. The held-out test split ignores this
  setting and keeps every negative frame (after striding), so your final
  reported metrics reflect the real positive/negative ratio rather than an
  artificially boosted one.

## 5. Train + evaluate

```bash
python train.py --config config.yaml
```

This will:
1. Discover videos, build the site-stratified train/val/test split, and list
   frame/annotation pairs for each.
2. Train with the backbone **frozen** for `freeze_backbone_epochs` epochs,
   then unfreeze and fine-tune the whole network at a lower learning rate.
3. Save the best checkpoint (by validation AP@0.5) to `checkpoints/best_model.pt`.
4. Save AP50/precision/recall/F1 curves to `outputs/training_curves.png`.
5. Load the best checkpoint and run one final evaluation on the held-out
   REAL-Colon test split, saving results to `outputs/test_results.json`.

Like the SegNet project, training auto-resumes from
`checkpoints/resume_checkpoint.pt` if interrupted (e.g. a Windows TDR
crash) - just rerun the same command. Set `resume: false` in the config to
force a clean restart instead.

## Metrics

- **AP@0.5** (Average Precision at IoU=0.5): the primary metric, COCO-style
  101-point interpolated, computed across all confidence thresholds.
- **Precision / Recall / F1**: computed at a single fixed operating point
  (`score_thresh` in the model, `score_thresh_for_prf1=0.5` inside
  `DetectionEvaluator` in `metrics.py`) - useful for a concrete
  "at this confidence cutoff, how many polyps did it catch / how many false
  alarms" read, alongside the threshold-independent AP50.
- A prediction counts as a match (True Positive) if its IoU with an
  unmatched ground-truth box is >= 0.5; each ground-truth box can only be
  matched once, so extra duplicate detections on an already-matched polyp
  count as false positives.

## Notes on design choices

- **Backbone**: ResNet-50 with a Feature Pyramid Network (FPN), pretrained
  on COCO (`retinanet_resnet50_fpn_v2`) - the classification head is
  replaced for the single "polyp" class, but the pretrained backbone/FPN
  weights are kept (transfer learning, same philosophy as the SegNet
  project's pretrained VGG-19 encoder).
- **Why freeze-then-unfreeze**: the classification/regression heads are
  freshly initialized (not pretrained), so training the whole network from
  step 1 risks the large early gradients from those random heads corrupting
  the pretrained backbone before the heads have learned anything sensible.
  A short frozen-backbone warm-up lets the heads catch up first.
- **Why site-stratified video-level splitting, not a random frame split**: a
  random split over individual frames would put near-duplicate consecutive
  frames of the same polyp into both train and test, making test metrics
  look far better than the model's true generalization - this is the same
  kind of leakage cross-dataset SegNet evaluation was designed to avoid, just
  at the frame/video level instead of the dataset level.
