"""
Hard-negative fine-tuning for scope/endoscope-only frames.

Goals:
  1. Continue from the existing best checkpoint without overwriting it.
  2. Scope-negative images DO NOT need mask files. Their all-zero masks are
     created in memory on the fly.
  3. Mix original polyp samples into training to reduce catastrophic forgetting.
  4. Save every fine-tuning run under a unique run directory, with:
       - a rollback copy of the source checkpoint
       - the best safe scope-finetuned checkpoint
       - the last checkpoint
       - history/metadata
  5. Load the same architecture used by the original training. If config and
     checkpoint disagree, detect the checkpoint family where possible and give
     a useful diagnostic rather than a huge unexplained state_dict error.
"""

import argparse
import json
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import (
    ZERO_MASK_SENTINEL,
    build_stratified_splits,
    get_train_transforms,
    get_val_transforms,
    list_scope_negative_pairs,
)
from metrics import BCEDiceLoss, dice_coeff, iou_score, pixel_accuracy
from model import build_model


FINETUNE_SUBDIR = "finetune_scope_negatives"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_run_id(run_name=None):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not run_name:
        return f"scope_negatives_{stamp}"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name).strip("_")
    return f"{safe or 'scope_negatives'}_{stamp}"


def _extract_state_dict(checkpoint):
    """Accept raw state_dict or common wrapped checkpoint formats."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")

    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            checkpoint = value
            break

    if not checkpoint:
        raise ValueError("Checkpoint is empty.")

    # Only remove a LEADING DataParallel/DDP prefix.
    clean = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            # A raw state_dict should be tensor-only. Ignore metadata entries if any.
            continue
        new_key = key[7:] if key.startswith("module.") else key
        clean[new_key] = value

    if not clean:
        raise ValueError("Could not find tensor weights in checkpoint.")
    return clean


def _detect_variant_from_state_dict(state):
    keys = set(state.keys())
    if any(k.startswith("net.segformer.") or k.startswith("net.decode_head.") for k in keys):
        return "segformer"
    if any(k.startswith("aspp.") or k.startswith("enc1_5x5.") or k.startswith("dec1_5x5.") for k in keys):
        return "modified"
    if any(k.startswith("enc1.") for k in keys) and any(k.startswith("dec5.") for k in keys):
        return "plain"
    return None



def _convert_segformer_state_dict_layout(state, model):
    """
    Convert SegFormer state_dict key names between Hugging Face Transformers
    layouts. Conversion is only applied when source and target layouts differ.
    strict=True is still required by the caller afterwards.
    """
    source_keys = list(state.keys())
    target_keys = list(model.state_dict().keys())

    source_new = any(".segformer.stages." in k for k in source_keys)
    source_legacy = any(".segformer.encoder.patch_embeddings." in k for k in source_keys)
    target_new = any(".segformer.stages." in k for k in target_keys)
    target_legacy = any(".segformer.encoder.patch_embeddings." in k for k in target_keys)

    if source_new and target_legacy:
        print("[INFO] Converting SegFormer checkpoint keys: newer Transformers layout -> legacy layout")

        def convert_key(k):
            k = re.sub(
                r"(\.segformer)\.stages\.(\d+)\.patch_embeddings\.",
                r"\1.encoder.patch_embeddings.\2.",
                k,
            )
            k = re.sub(
                r"(\.segformer)\.stages\.(\d+)\.blocks\.(\d+)\.",
                r"\1.encoder.block.\2.\3.",
                k,
            )
            k = re.sub(
                r"(\.segformer)\.stages\.(\d+)\.layer_norm\.",
                r"\1.encoder.layer_norm.\2.",
                k,
            )

            k = k.replace(".layernorm_before.", ".layer_norm_1.")
            k = k.replace(".layernorm_after.", ".layer_norm_2.")
            k = k.replace(".attention.q_proj.", ".attention.self.query.")
            k = k.replace(".attention.k_proj.", ".attention.self.key.")
            k = k.replace(".attention.v_proj.", ".attention.self.value.")
            k = k.replace(
                ".attention.sequence_reduction.sequence_reduction.",
                ".attention.self.sr.",
            )
            k = k.replace(
                ".attention.sequence_reduction.layer_norm.",
                ".attention.self.layer_norm.",
            )
            k = k.replace(".attention.o_proj.", ".attention.output.dense.")
            k = k.replace(".mlp.fc1.", ".mlp.dense1.")
            k = k.replace(".mlp.fc2.", ".mlp.dense2.")
            k = k.replace(".decode_head.linear_projections.", ".decode_head.linear_c.")
            return k

        converted = {}
        for k, v in state.items():
            nk = convert_key(k)
            if nk in converted:
                raise RuntimeError(f"Duplicate key after SegFormer conversion: {nk}")
            converted[nk] = v
        return converted

    if source_legacy and target_new:
        print("[INFO] Converting SegFormer checkpoint keys: legacy Transformers layout -> newer layout")

        def convert_key(k):
            k = re.sub(
                r"(\.segformer)\.encoder\.patch_embeddings\.(\d+)\.",
                r"\1.stages.\2.patch_embeddings.",
                k,
            )
            k = re.sub(
                r"(\.segformer)\.encoder\.block\.(\d+)\.(\d+)\.",
                r"\1.stages.\2.blocks.\3.",
                k,
            )
            k = re.sub(
                r"(\.segformer)\.encoder\.layer_norm\.(\d+)\.",
                r"\1.stages.\2.layer_norm.",
                k,
            )

            k = k.replace(".layer_norm_1.", ".layernorm_before.")
            k = k.replace(".layer_norm_2.", ".layernorm_after.")
            k = k.replace(".attention.self.query.", ".attention.q_proj.")
            k = k.replace(".attention.self.key.", ".attention.k_proj.")
            k = k.replace(".attention.self.value.", ".attention.v_proj.")
            k = k.replace(
                ".attention.self.sr.",
                ".attention.sequence_reduction.sequence_reduction.",
            )
            k = k.replace(
                ".attention.self.layer_norm.",
                ".attention.sequence_reduction.layer_norm.",
            )
            k = k.replace(".attention.output.dense.", ".attention.o_proj.")
            k = k.replace(".mlp.dense1.", ".mlp.fc1.")
            k = k.replace(".mlp.dense2.", ".mlp.fc2.")
            k = k.replace(".decode_head.linear_c.", ".decode_head.linear_projections.")
            return k

        converted = {}
        for k, v in state.items():
            nk = convert_key(k)
            if nk in converted:
                raise RuntimeError(f"Duplicate key after SegFormer conversion: {nk}")
            converted[nk] = v
        return converted

    return state


def _build_and_load_model(cfg, ft_cfg, state, device):
    configured_variant = str(ft_cfg.get("model_variant", cfg.get("model_variant", "modified"))).lower()
    detected_variant = _detect_variant_from_state_dict(state)

    # Prefer what the checkpoint itself tells us. This fixes the common case where
    # training used SegFormer but the fine-tune script was hard-coded to "modified".
    model_variant = detected_variant or configured_variant
    segformer_size = str(ft_cfg.get("segformer_size", cfg.get("segformer_size", "b3"))).lower()
    num_classes = int(cfg.get("num_classes", 1))

    if detected_variant and detected_variant != configured_variant:
        print(
            f"[WARN] config requests model_variant='{configured_variant}', but checkpoint "
            f"looks like '{detected_variant}'. Using checkpoint-detected variant '{detected_variant}'."
        )

    print(f"Building model for checkpoint: variant={model_variant}, segformer_size={segformer_size}")
    model = build_model(
        model_variant,
        num_classes=num_classes,
        pretrained=False,
        segformer_size=segformer_size,
    ).to(device)

    # SegFormer internal module names changed across Transformers versions.
    # Translate only when needed, then still demand a strict load.
    state_for_model = (
        _convert_segformer_state_dict_layout(state, model)
        if model_variant == "segformer"
        else state
    )

    try:
        model.load_state_dict(state_for_model, strict=True)
        if state_for_model is not state:
            print("[OK] SegFormer checkpoint key layout converted and loaded strictly.")
    except RuntimeError as exc:
        model_keys = set(model.state_dict().keys())
        state_keys = set(state_for_model.keys())
        missing = sorted(model_keys - state_keys)
        unexpected = sorted(state_keys - model_keys)
        shape_mismatch = []
        for key in sorted(model_keys & state_keys):
            if tuple(model.state_dict()[key].shape) != tuple(state_for_model[key].shape):
                shape_mismatch.append(
                    f"{key}: checkpoint={tuple(state_for_model[key].shape)} model={tuple(model.state_dict()[key].shape)}"
                )

        details = [
            "Checkpoint could not be loaded strictly.",
            f"Configured variant: {configured_variant}",
            f"Detected variant: {detected_variant}",
            f"Attempted variant: {model_variant}",
            f"SegFormer size: {segformer_size}",
        ]
        if missing:
            details.append(f"Missing keys (first 12): {missing[:12]}")
        if unexpected:
            details.append(f"Unexpected keys (first 12): {unexpected[:12]}")
        if shape_mismatch:
            details.append(f"Shape mismatches (first 12): {shape_mismatch[:12]}")
        if model_variant == "segformer":
            details.append(
                "If the original checkpoint is SegFormer, make sure config.yaml has the SAME "
                "segformer_size used in train.py (for example b3)."
            )
        raise RuntimeError("\n".join(details)) from exc

    return model, model_variant, segformer_size


class MixedPolypDataset(Dataset):
    """
    Supports:
      - normal polyp samples: (image_path, real_mask_path, dataset_name)
      - scope negatives:      (image_path, ZERO_MASK_SENTINEL, "scope_negative")

    Scope-negative masks are generated in RAM and never need to exist on disk.
    A missing/corrupt REAL polyp mask is an error; it is never converted to a
    negative target silently.
    """

    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        max_retries = 5
        for attempt in range(max_retries):
            item = self.pairs[idx]
            img_path = item[0]
            mask_path = item[1]
            ds_name = item[2] if len(item) > 2 else "unknown"

            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"[WARN] Unreadable image, retrying: {img_path}")
                idx = np.random.randint(0, len(self.pairs))
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            if mask_path is None or mask_path == ZERO_MASK_SENTINEL:
                # Explicit hard negative: create target in memory only.
                mask = np.zeros((h, w), dtype=np.float32)
            else:
                mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    raise FileNotFoundError(
                        f"Real training mask could not be read: {mask_path}\n"
                        "This sample will NOT be treated as a negative. Fix/remove the bad pair."
                    )
                mask = (mask_img > 127).astype(np.float32)

            if self.transform is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            return image, mask.float(), ds_name

        raise RuntimeError(f"Could not obtain a readable image after {max_retries} attempts.")


def run_epoch(model, loader, criterion, optimizer, device, train, scaler, use_amp):
    model.train() if train else model.eval()
    total_loss = total_dice = total_iou = total_acc = 0.0
    n_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc="train" if train else "eval", leave=False)
        for batch in pbar:
            images, masks = batch[0].to(device), batch[1].to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, masks)

            if train:
                if use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            probs = torch.sigmoid(logits.float())
            total_loss += loss.item()
            total_dice += dice_coeff(probs, masks).item()
            total_iou += iou_score(probs, masks).item()
            total_acc += pixel_accuracy(probs, masks).item()
            n_batches += 1
            pbar.set_postfix(loss=loss.item())

    denom = max(n_batches, 1)
    return total_loss / denom, total_dice / denom, total_iou / denom, total_acc / denom


def compute_scope_fp_rate(model, negative_pairs, image_size, device, use_amp, threshold=0.5):
    """Mean fraction of pixels incorrectly predicted as polyp on scope-only frames."""
    ds = MixedPolypDataset(negative_pairs, transform=get_val_transforms(image_size))
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    model.eval()
    total_fp_frac, n = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                logits = model(images)
            probs = torch.sigmoid(logits.float())
            pred_bin = (probs > threshold).float()
            frac_per_image = pred_bin.reshape(pred_bin.size(0), -1).mean(dim=1)
            total_fp_frac += frac_per_image.sum().item()
            n += images.size(0)

    return total_fp_frac / max(n, 1)


def main(cfg_path, source_checkpoint=None, run_name=None):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key in ["lr", "lr_finetune", "weight_decay", "train_fraction", "val_fraction", "test_fraction"]:
        if key in cfg:
            cfg[key] = float(cfg[key])
    for key in [
        "image_size", "batch_size", "epochs", "num_workers", "seed",
        "early_stopping_patience", "freeze_encoder_epochs", "in_channels", "num_classes",
    ]:
        if key in cfg:
            cfg[key] = int(cfg[key])

    if "finetune_negatives" not in cfg:
        raise KeyError("config.yaml must contain a 'finetune_negatives' section.")
    ft_cfg = cfg["finetune_negatives"]

    image_size = int(cfg["image_size"])
    use_amp = bool(cfg.get("use_amp", True))
    ft_epochs = int(ft_cfg.get("epochs", 15))
    ft_lr = float(ft_cfg.get("lr", 1e-6))
    ft_batch_size = int(ft_cfg.get("batch_size", cfg["batch_size"]))
    positive_mix_ratio = float(ft_cfg.get("positive_mix_ratio", 1.0))
    max_val_dice_drop = float(ft_cfg.get("max_val_dice_drop", 0.02))
    patience = int(ft_cfg.get("patience", 3))
    fp_threshold = float(ft_cfg.get("fp_threshold", 0.5))

    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | AMP: {use_amp and device.type == 'cuda'}")

    orig_ckpt_dir = Path(cfg["checkpoint_dir"])
    source_ckpt_path = Path(source_checkpoint) if source_checkpoint else Path(
        ft_cfg.get("source_checkpoint", orig_ckpt_dir / "best_model.pt")
    )
    if not source_ckpt_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_ckpt_path}")

    run_id = _safe_run_id(run_name or ft_cfg.get("run_name"))
    run_ckpt_dir = orig_ckpt_dir / FINETUNE_SUBDIR / run_id
    run_out_dir = Path(cfg["output_dir"]) / FINETUNE_SUBDIR / run_id
    run_ckpt_dir.mkdir(parents=True, exist_ok=False)
    run_out_dir.mkdir(parents=True, exist_ok=False)

    best_ckpt_path = run_ckpt_dir / f"best_{run_id}.pt"
    last_ckpt_path = run_ckpt_dir / f"last_{run_id}.pt"
    rollback_ckpt_path = run_ckpt_dir / f"rollback_source_{source_ckpt_path.name}"
    shutil.copy2(source_ckpt_path, rollback_ckpt_path)

    print(f"\nRun ID: {run_id}")
    print(f"Source checkpoint: {source_ckpt_path}")
    print(f"Rollback copy:     {rollback_ckpt_path}")
    print(f"Best fine-tune:    {best_ckpt_path}")

    # Same stratified source splits as train.py.
    train_pairs, val_pairs, _ = build_stratified_splits(cfg)

    # No mask files are required for these frames.
    negative_pairs = list_scope_negative_pairs(
        ft_cfg["images_dir"], ft_cfg.get("img_ext", ".bmp")
    )
    # Be backward-compatible with older dataset.py versions that returned None.
    negative_pairs = [
        (img, ZERO_MASK_SENTINEL if mask is None else mask, name)
        for img, mask, name in negative_pairs
    ]
    print(f"Loaded {len(negative_pairs)} scope-negative images; zero masks will be generated in memory.")

    n_positives_to_mix = min(len(train_pairs), int(round(len(negative_pairs) * positive_mix_ratio)))
    rng = random.Random(int(cfg["seed"]))
    positive_sample = rng.sample(train_pairs, n_positives_to_mix) if n_positives_to_mix else []
    print(f"Mixed in {len(positive_sample)} original samples (positive_mix_ratio={positive_mix_ratio}).")

    combined_train_pairs = negative_pairs + positive_sample
    rng.shuffle(combined_train_pairs)

    train_loader = DataLoader(
        MixedPolypDataset(combined_train_pairs, transform=get_train_transforms(image_size)),
        batch_size=ft_batch_size,
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        MixedPolypDataset(val_pairs, transform=get_val_transforms(image_size)),
        batch_size=ft_batch_size,
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )

    checkpoint_obj = torch.load(source_ckpt_path, map_location=device, weights_only=False)
    state = _extract_state_dict(checkpoint_obj)
    model, model_variant, segformer_size = _build_and_load_model(cfg, ft_cfg, state, device)
    print(f"[OK] Loaded source weights successfully ({model_variant}).")

    model.set_encoder_trainable(True)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=ft_lr, weight_decay=float(cfg.get("weight_decay", 0.0))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    print("\nMeasuring baseline before hard-negative fine-tuning...")
    baseline_loss, baseline_dice, baseline_iou, baseline_acc = run_epoch(
        model, val_loader, criterion, optimizer, device, train=False, scaler=scaler, use_amp=use_amp
    )
    baseline_fp_rate = compute_scope_fp_rate(
        model, negative_pairs, image_size, device, use_amp, threshold=fp_threshold
    )
    print(
        f"Baseline Val Dice: {baseline_dice:.4f} | Val IoU: {baseline_iou:.4f} | "
        f"Scope FP pixel rate: {baseline_fp_rate:.6f}"
    )

    # A candidate is safe only if it stays inside the allowed validation Dice drop.
    # Among safe candidates, prefer LOWER scope false-positive rate. If tied, prefer
    # higher validation Dice. This directly optimizes the purpose of this fine-tune.
    best_safe_fp = baseline_fp_rate
    best_safe_dice = baseline_dice
    saved_best = False
    epochs_below_guardrail = 0
    history = {
        "run_id": run_id,
        "source_checkpoint": str(source_ckpt_path),
        "rollback_checkpoint": str(rollback_ckpt_path),
        "model_variant": model_variant,
        "segformer_size": segformer_size,
        "baseline": {
            "val_loss": baseline_loss,
            "val_dice": baseline_dice,
            "val_iou": baseline_iou,
            "val_accuracy": baseline_acc,
            "scope_fp_rate": baseline_fp_rate,
        },
        "epochs": [],
    }

    for epoch in range(1, ft_epochs + 1):
        tr_loss, tr_dice, tr_iou, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            train=True, scaler=scaler, use_amp=use_amp,
        )
        val_loss, val_dice, val_iou, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device,
            train=False, scaler=scaler, use_amp=use_amp,
        )
        fp_rate = compute_scope_fp_rate(
            model, negative_pairs, image_size, device, use_amp, threshold=fp_threshold
        )

        drop = baseline_dice - val_dice
        safe = drop <= max_val_dice_drop
        improves_scope = fp_rate < best_safe_fp - 1e-12
        ties_scope_better_dice = abs(fp_rate - best_safe_fp) <= 1e-12 and val_dice > best_safe_dice

        print(
            f"Epoch {epoch:2d}/{ft_epochs:2d} | "
            f"Train loss={tr_loss:.4f} dice={tr_dice:.4f} | "
            f"Val dice={val_dice:.4f} drop={drop:+.4f} | "
            f"Scope FP={fp_rate:.6f} | safe={safe}"
        )

        torch.save(model.state_dict(), last_ckpt_path)

        if safe and (improves_scope or ties_scope_better_dice):
            best_safe_fp = fp_rate
            best_safe_dice = val_dice
            torch.save(model.state_dict(), best_ckpt_path)
            saved_best = True
            print(
                f"  -> NEW BEST SAFE checkpoint: scope FP {best_safe_fp:.6f}, "
                f"val Dice {best_safe_dice:.4f}"
            )

        history["epochs"].append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_dice": tr_dice,
            "train_iou": tr_iou,
            "train_accuracy": tr_acc,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "val_accuracy": val_acc,
            "val_dice_drop_from_baseline": drop,
            "scope_fp_rate": fp_rate,
            "safe": safe,
        })

        if safe:
            epochs_below_guardrail = 0
        else:
            epochs_below_guardrail += 1
            if epochs_below_guardrail >= patience:
                print(
                    f"\n[EARLY STOPPING] Validation Dice exceeded the allowed drop "
                    f"({max_val_dice_drop}) for {patience} consecutive epochs."
                )
                break

    history["best_safe"] = {
        "saved": saved_best,
        "checkpoint": str(best_ckpt_path) if saved_best else None,
        "val_dice": best_safe_dice if saved_best else baseline_dice,
        "scope_fp_rate": best_safe_fp if saved_best else baseline_fp_rate,
    }
    history["last_checkpoint"] = str(last_ckpt_path)

    with open(run_out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(run_ckpt_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "source_checkpoint": str(source_ckpt_path),
            "rollback_checkpoint": str(rollback_ckpt_path),
            "best_checkpoint": str(best_ckpt_path) if saved_best else None,
            "last_checkpoint": str(last_ckpt_path),
            "model_variant": model_variant,
            "segformer_size": segformer_size,
            "scope_images": len(negative_pairs),
            "mixed_original_samples": len(positive_sample),
        }, f, indent=2)

    print("\n" + "=" * 90)
    print("Fine-tuning completed.")
    print(f"Original source is UNTOUCHED: {source_ckpt_path}")
    print(f"Rollback copy:                {rollback_ckpt_path}")
    if saved_best:
        print(f"Best safe scope checkpoint:   {best_ckpt_path}")
        print(f"  Val Dice: {best_safe_dice:.4f} | Scope FP rate: {best_safe_fp:.6f}")
    else:
        print("No fine-tuned epoch beat the baseline scope FP rate while satisfying the Dice guardrail.")
        print("Use the rollback/source checkpoint; it remains unchanged.")
    print(f"Last epoch checkpoint:        {last_ckpt_path}")
    print(f"History:                     {run_out_dir / 'history.json'}")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--source-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint to continue from. Defaults to finetune_negatives.source_checkpoint or checkpoint_dir/best_model.pt.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional readable run prefix. A timestamp is always appended to keep it unique.",
    )
    args = parser.parse_args()
    main(args.config, source_checkpoint=args.source_checkpoint, run_name=args.run_name)
