# Polyp characterization after segmentation

ResNet-18 initialized from ImageNet predicts the spreadsheet's four JNET labels:
`1`, `2A`, `2B`, `3`. This is an experimental optical characterization model,
not a pathology diagnosis model. Inputs are RGB crops with 15% context around
connected components of the existing SegFormer-B3 segmentation mask.

## Train

From the repository root, using the existing segmentation Python environment:

```powershell
python 'Polyps Characterization/pipeline.py' train
```

Defaults to the user's ERCPMP directory and
`Polyps Segmentation/checkpoints/best_model.pt`. Requires torch, torchvision,
numpy, Pillow, opencv-python, scikit-learn, transformers, PyYAML and the existing
segmentation module dependencies. The XLSX reader uses Python's standard library.
ImageNet ResNet-18 weights and the SegFormer-B3 configuration must be cached or
downloadable. Training refuses to overwrite an existing `best.pt`; choose a new
`--output` directory for another experiment.

Only still JPG images with an explicit valid JNET entry in spreadsheet column L
are included. Missing labels are not inferred from filenames or adjacent columns.
Videos are not sampled. Patient codes group all frames into one split, with
approximately 60/20/20 train/validation/test allocation within each JNET class.
The seed is 42. Label exclusions are recorded in `excluded.json`.

The frozen segmentation model generates crops before classification training.
The largest component supplies the crop for each training image because labels
are available at patient/image level, not separately for each component. Images
with no component of at least 64 pixels are skipped and counted. Evaluation
therefore measures characterization conditional on successful segmentation;
coverage is reported separately. Segmentation has no ERCPMP ground-truth masks
here, so localization accuracy cannot be measured by this run.

Training balances sampling by class and patient so patients with more images do
not dominate. The backbone is frozen for three epochs, then fine-tuned at 1e-5
while the classification head uses 3e-4. BatchNorm running statistics stay frozen.
Training runs for up to 40 epochs with patience 10, selecting the best checkpoint
by validation patient macro F1. Patient predictions average image probabilities.
The test set is evaluated once after selection.

## Run both layers

```powershell
python 'Polyps Characterization/pipeline.py' predict --image 'path/to/frame.jpg'
```

To reuse an existing full-resolution binary mask:

```powershell
python 'Polyps Characterization/pipeline.py' predict --image 'path/to/frame.jpg' --mask 'path/to/mask.png'
```

`Characterizer(checkpoint, device).predict(rgb, mask)` is the integration API.
RGB must be uint8 H×W×3; mask must be a same-size H×W binary array. Each qualifying
component receives a bounding box, JNET label, and four uncalibrated softmax
scores. Empty masks return an empty list. These scores are not calibrated clinical
confidence. Multiple-component inference is supported, but training labels only
supervise the largest component and do not validate multi-polyp attribution.

Outputs in `outputs/ercpmp_jnet/` include `best.pt`, `manifest.json`,
`data_summary.json`, `history.json`, `test_results.json`, and cached crops.
The checkpoint records the segmentation checkpoint path and SHA-256. Use the
same segmentation weights and preprocessing at inference. Changing the first
layer requires a new evaluation. Patient test counts are small, and there is no
external validation; this model is for research evaluation.
