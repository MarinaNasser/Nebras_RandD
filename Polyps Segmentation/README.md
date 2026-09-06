# Polyp Segmentation: SegNet (VGG-19 encoder) — Cross-Dataset Evaluation

Trains a SegNet whose encoder is a VGG-19 (batchnorm) backbone **pretrained on ImageNet**
(transfer learning — the encoder is never trained from scratch), on all but one of
your polyp datasets, and validates on the **held-out** dataset. Decoder weights are
trained from scratch (as in the original SegNet paper) since no pretrained decoder exists.

Datasets supported out of the box: **CVC-ClinicDB**, **CVC-ColonDB**, **ETIS-LaribPolypDB**,
**Kvasir-SEG** — and you can add more. Any number of datasets ≥ 2 can be listed in
`config.yaml`; all of them except the one named `held_out_dataset` are pooled together
for training, so adding a 5th, 6th, etc. dataset is just adding another block to
`config.yaml`'s `datasets` section — no code changes needed.

## 1. Install

```bash
pip install -r requirements.txt
```

GPU strongly recommended (SegNet + VGG-19 is heavy). Works on CPU but will be slow.

## Interactive Gradio inference demo

After installing the requirements, launch the browser-based demo with:

```bash
python gradio_app.py
```

Then open `http://127.0.0.1:7860`. Upload an endoscopy frame to see the contour
overlay, binary segmentation mask, probability heatmap, confidence, and runtime.
Use `--share` to request a temporary public Gradio link, or select a different
checkpoint with `--checkpoint path/to/model.pt`.

## Validate on external datasets

Scan all top-level datasets in `C:\Users\Omen Max\Datasets`, automatically skip
the training datasets listed in `config.yaml`, and benchmark the trained model:

```bash
python validate_external_datasets.py
```

The command writes `external_validation.json` and `external_validation.csv` under
`outputs/external_validation`. Raster segmentation masks are paired automatically
when available; datasets with only image-level or detection annotations receive
timing results but no Dice/IoU. For a quick smoke test, use `--max-images 10`.
To test exactly 100 images distributed across the eligible datasets, use
`--total-images 100`; selection is reproducible with `--seed` (default: 42).

## 2. Point the config at your data

Open `config.yaml` and fill in the real `images_dir` / `masks_dir` / extensions for
each dataset on your machine. Folder names differ depending on where you downloaded
each dataset from — this is the only file you should need to edit for a standard run.

Set `held_out_dataset` to whichever dataset you want to validate on (train
automatically pools every other listed dataset).

If your masks don't share filename stems with their images (e.g. `23.tif` image vs.
`mask_23.tif` mask, or numbering offset by source), adjust
`match_mask_for_image()` in `dataset.py` — it's isolated there specifically so you
don't have to touch anything else.

## 3. Train + evaluate

```bash
python train.py --config config.yaml
```

This will:
1. Pool the 2 training datasets, carve out a small internal validation slice for
   early stopping (drawn only from the training pool — the held-out dataset is
   never touched during training).
2. Train with the encoder **frozen** for `freeze_encoder_epochs` epochs (default 5),
   then unfreeze it and fine-tune the whole network at a lower learning rate.
3. Save the best checkpoint (by internal validation Dice) to `checkpoints/best_model.pt`.
4. Save loss/Dice/IoU curves to `outputs/training_curves.png`.
5. Load the best checkpoint and run one final evaluation on the **held-out
   dataset**, saving Dice/IoU/loss to `outputs/test_results.json`.

## 4. Look at predictions qualitatively

```bash
python visualize_predictions.py --config config.yaml --n 8
```

Saves a grid of (image / ground truth / prediction) for a random sample of the
held-out test set to `outputs/qualitative_predictions.png`.

## Where the metrics are, and how to see them

`train.py` prints a formatted summary table (per-epoch train/val loss, Dice, IoU,
plus the final held-out test Dice/IoU) automatically right after training finishes.
Everything is also saved to disk as it goes, so you're never locked out of it:

- `outputs/history.json` — every epoch's train/val loss, Dice, IoU (updated after each epoch)
- `outputs/test_results.json` — the final held-out test set metrics (written once, at the end)
- `outputs/training_curves.png` — loss/Dice/IoU curves over training

To re-print (or re-generate) that summary at any time — after training finishes,
or even mid-training to check progress so far — without retraining or needing a GPU:

```bash
python show_results.py --config config.yaml          # prints the summary table
python show_results.py --config config.yaml --plot   # also re-saves training_curves.png
```

## 5. Full cross-dataset evaluation (every dataset held out once)

To get a result for every dataset as the held-out set, rerun with `held_out_dataset`
changed each time — nothing else in the code needs to change. With 4 datasets
configured (CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB, Kvasir-SEG), each run
trains on the other 3 pooled together:

```bash
for ds in CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB Kvasir-SEG; do
  python - <<PY
import yaml
cfg = yaml.safe_load(open("config.yaml"))
cfg["held_out_dataset"] = "$ds"
yaml.safe_dump(cfg, open("config.yaml", "w"))
PY
  python train.py --config config.yaml
  mv checkpoints/best_model.pt checkpoints/best_model_${ds}.pt
  mv outputs/test_results.json outputs/test_results_${ds}.json
done
```

Note ETIS-LaribPolypDB is the smallest and hardest of the three CVC/ETIS sets (small,
flat polyps, different scope/lighting) — expect noticeably lower Dice/IoU when it's
the held-out set, which is consistent with published benchmarks. Kvasir-SEG images
are generally higher-resolution and more visually distinct (different capture
hardware/site), so mixing it in as a training source can help generalization, but
holding it out as the test set is also a reasonably hard cross-domain check.

## Troubleshooting: "CUDA error: the launch timed out and was terminated" (Windows)

This is **Windows' TDR (Timeout Detection and Recovery) watchdog**, not a bug in this
code or an out-of-memory error. Windows assumes any single GPU kernel running longer
than ~2 seconds means the display driver has hung, and force-resets it — this hits
anyone doing GPU compute on their main display GPU on Windows, especially with a
slower GPU or a heavier model.

**If you have admin rights**, the direct fix is raising the timeout: open
`regedit` → `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` →
create a DWORD `TdrDelay` set to `60` (seconds) → restart. This is the standard fix
recommended for ML training on Windows.

**Without admin rights**, use the code-side mitigations already built into this
project:
- `use_amp: true` in `config.yaml` (on by default) — mixed precision roughly halves
  per-kernel GPU time on most NVIDIA GPUs, directly reducing the odds any one kernel
  crosses the TDR limit.
- Smaller `image_size` / `batch_size` — smaller tensors mean smaller, faster kernel
  launches. Defaults are now `image_size: 256`, `batch_size: 4`; drop further
  (e.g. `image_size: 192`, `batch_size: 2`) if crashes persist.
- **Auto-resume**: if training does still crash, just rerun `python train.py`. It
  automatically picks up from `checkpoints/resume_checkpoint.pt`, which is saved
  every `save_every_n_steps` steps (default 50) as well as at the end of every
  epoch — so a crash costs you at most ~50 steps of progress, not the whole run.
  Set `resume: false` in the config if you ever want to force a clean restart instead.

## Notes on design choices

- **Model variant**: `config.yaml`'s `model_variant` defaults to `"modified"` — this adds
  three things on top of plain SegNet, following the polyp-segmentation SegNet literature:
  1. **Skip connections** at all 5 encoder/decoder depths (concatenated, then convolved
     down). Plain SegNet only passes *pooling indices* to the decoder, not feature
     content, so fine boundary detail is genuinely lost; skip connections fix that,
     much like U-Net.
  2. **Extra 5x5 conv blocks** at the first two encoder depths and last two decoder
     depths, to suppress the background noise that small 3x3 filters pick up from
     colon-lining texture that superficially resembles polyp texture.
  3. **A parallel dilated-convolution bottleneck** (rates 1/6/12/18, ASPP-style),
     giving the network multi-scale context right where the feature map is smallest -
     this specifically helps recover small/flat polyps that plain SegNet's linear
     pool-heavy downsampling path tends to lose.
  Set `model_variant: "plain"` if you want the original SegNet as a baseline to
  compare against — both share the same VGG19_bn pretrained encoder loading path.
  The modified variant has ~50M params vs. ~40M for plain, and needs somewhat more
  VRAM (skip connections keep full-resolution feature maps alive longer during the
  decoder pass) - drop `batch_size` first if you hit OOM with it.
- **Why VGG19_bn and not plain VGG19**: batchnorm gives more stable training when
  fine-tuning on a small (~1000-image) medical dataset; torchvision ships
  ImageNet-pretrained weights for `vgg19_bn` used here.
- **Why freeze-then-unfreeze**: training the whole network from step 1 on ~1,000
  images tends to overwrite the useful pretrained low-level filters before the new
  decoder has learned anything sensible. A short frozen-encoder warmup lets the
  decoder catch up first.
- **Loss**: BCE + Dice combo — Dice alone can be unstable early in training when
  predictions are near-random; BCE stabilizes early epochs, Dice pushes final
  overlap quality (important given polyps are a small foreground fraction of
  most frames — class imbalance).
- **image_size 384**: matches typical resolutions used in CVC-ClinicDB/ColonDB/ETIS
  papers reasonably well and keeps memory footprint manageable on a single consumer
  GPU. Lower to 256 if you hit out-of-memory errors (SegNet's stored pooling
  indices, and the modified variant's skip connections, both add memory overhead
  beyond a typical U-Net at the same resolution).
