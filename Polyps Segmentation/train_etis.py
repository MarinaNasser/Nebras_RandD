"""
train_etis.py
Trains and evaluates exclusively on the ETIS-LaribPolypDB dataset.
Splits ETIS-LaribPolypDB into train, val, and test splits (e.g. 70/10/20).
"""
import argparse
import random
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PolypDataset, list_pairs, get_train_transforms, get_val_transforms
from model import build_model
from metrics import BCEDiceLoss, dice_coeff, iou_score, pixel_accuracy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_batch_metrics(pred_probs, targets, eps=1e-7):
    p_bin = (pred_probs > 0.5).float()
    p_flat = p_bin.reshape(pred_probs.size(0), -1)
    t_flat = targets.reshape(targets.size(0), -1)

    inter = (p_flat * t_flat).sum(dim=1)
    union = p_flat.sum(dim=1) + t_flat.sum(dim=1)
    dices = ((2.0 * inter + eps) / (union + eps)).cpu().tolist()

    union_iou = union - inter
    ious = ((inter + eps) / (union_iou + eps)).cpu().tolist()

    precisions = ((inter + eps) / (p_flat.sum(dim=1) + eps)).cpu().tolist()
    recalls = ((inter + eps) / (t_flat.sum(dim=1) + eps)).cpu().tolist()

    return dices, ious, precisions, recalls


def run_epoch(model, loader, criterion, optimizer, device, train, scaler, use_amp):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    all_dices, all_ious, all_precs, all_recs = [], [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc="train" if train else "eval", leave=False)
        for images, masks, _ in pbar:
            images, masks = images.to(device), masks.to(device)

            if train:
                optimizer.zero_grad()

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
            n_batches += 1

            dices, ious, precs, recs = compute_batch_metrics(probs, masks)
            all_dices.extend(dices)
            all_ious.extend(ious)
            all_precs.extend(precs)
            all_recs.extend(recs)

            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / max(n_batches, 1)
    avg_dice = float(np.mean(all_dices)) if all_dices else 0.0
    avg_iou = float(np.mean(all_ious)) if all_ious else 0.0
    avg_prec = float(np.mean(all_precs)) if all_precs else 0.0
    avg_rec = float(np.mean(all_recs)) if all_recs else 0.0

    return avg_loss, avg_dice, avg_iou, avg_prec, avg_rec


def build_single_dataset_splits(cfg, target_dataset="ETIS-LaribPolypDB"):
    if target_dataset not in cfg["datasets"]:
        raise KeyError(f"Dataset '{target_dataset}' not defined in config.yaml")

    d = cfg["datasets"][target_dataset]
    pairs = list_pairs(d["images_dir"], d["masks_dir"], d["img_ext"], d["mask_ext"], dataset_name=target_dataset)

    train_frac = cfg.get("train_fraction", 0.70)
    val_frac = cfg.get("val_fraction", 0.10)
    rng = np.random.RandomState(cfg["seed"])
    rng.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    print(f"\n[{target_dataset}] Total Samples: {n_total}")
    print(f"Split Breakdown -> Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}\n")

    return train_pairs, val_pairs, test_pairs


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    for key in ["lr", "lr_finetune", "weight_decay", "train_fraction", "val_fraction", "test_fraction"]:
        if key in cfg:
            cfg[key] = float(cfg[key])
    for key in ["image_size", "batch_size", "epochs", "num_workers", "seed",
                "early_stopping_patience", "freeze_encoder_epochs", "in_channels", "num_classes"]:
        cfg[key] = int(cfg[key])

    use_amp = bool(cfg.get("use_amp", True))
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | AMP: {use_amp and device.type == 'cuda'}")

    ckpt_dir = Path(cfg["checkpoint_dir"]) / "etis_only"
    out_dir = Path(cfg["output_dir"]) / "etis_only"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    train_pairs, val_pairs, test_pairs = build_single_dataset_splits(cfg, target_dataset="ETIS-LaribPolypDB")
    image_size = cfg["image_size"]

    train_loader = DataLoader(
        PolypDataset(train_pairs, transform=get_train_transforms(image_size)),
        batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        PolypDataset(val_pairs, transform=get_val_transforms(image_size)),
        batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True
    )
    test_loader = DataLoader(
        PolypDataset(test_pairs, transform=get_val_transforms(image_size)),
        batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True
    )

    # ---- Model ----
    model = build_model(
        cfg.get("model_variant", "modified"),
        num_classes=cfg["num_classes"],
        pretrained=True,
        segformer_size=cfg.get("segformer_size", "b0")
    ).to(device)

    criterion = BCEDiceLoss(bce_weight=0.5)
    # 1. Fix GradScaler warning
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    # 2. Add prec/rec keys to history
    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_iou": [], "val_iou": [],
        "train_prec": [], "val_prec": [],
        "train_rec": [], "val_rec": [],
        "per_dataset": []
    }

    best_val_dice = -1.0
    epochs_no_improve = 0
    encoder_unfrozen = False

    model.set_encoder_trainable(False)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    # ---- Training Loop ----
    for epoch in range(1, cfg["epochs"] + 1):
        if epoch == cfg["freeze_encoder_epochs"] + 1 and not encoder_unfrozen:
            print(f"\n>>> Unfreezing encoder for fine-tuning at epoch {epoch}")
            model.set_encoder_trainable(True)
            encoder_unfrozen = True
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr_finetune"], weight_decay=cfg["weight_decay"])

        # Unpack all 6 returned values (including precision & recall)
        train_loss, train_dice, train_iou, train_prec, train_rec, train_ds_res = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True, scaler=scaler, use_amp=use_amp
        )
        val_loss, val_dice, val_iou, val_prec, val_rec, val_ds_res = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False, scaler=scaler, use_amp=use_amp
        )

        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice); history["val_dice"].append(val_dice)
        history["train_iou"].append(train_iou); history["val_iou"].append(val_iou)
        history["train_prec"].append(train_prec); history["val_prec"].append(val_prec)
        history["train_rec"].append(train_rec); history["val_rec"].append(val_rec)
        history["per_dataset"].append({"epoch": epoch, "train": train_ds_res, "val": val_ds_res})

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"train_loss: {train_loss:.4f} dice: {train_dice:.4f} iou: {train_iou:.4f} prec: {train_prec:.4f} rec: {train_rec:.4f} | "
              f"val_loss: {val_loss:.4f} dice: {val_dice:.4f} iou: {val_iou:.4f} prec: {val_prec:.4f} rec: {val_rec:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            print(f"  -> new best overall val Dice {best_val_dice:.4f}, checkpoint saved")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg["early_stopping_patience"]:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ---- Final Evaluation on Train, Val, and Test Sets ----
    print("\nLoading best checkpoint for final evaluation on ETIS-LaribPolypDB...")
    model.load_state_dict(torch.load(ckpt_dir / "best_model.pt", map_location=device))

    eval_train_loader = DataLoader(
        PolypDataset(train_pairs, transform=get_val_transforms(image_size)),
        batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True
    )

    splits = [
        ("Train", eval_train_loader, len(train_pairs)),
        ("Val", val_loader, len(val_pairs)),
        ("Test", test_loader, len(test_pairs)),
    ]

    print("\n" + "=" * 80)
    print(f"{'DATASET':<20}{'SPLIT':<10}{'DICE':<12}{'IOU':<12}{'PRECISION':<12}{'RECALL':<12}{'COUNT':<8}")
    print("=" * 80)
    for split_name, loader, count in splits:
        loss, dice, iou, prec, rec = run_epoch(
            model, loader, criterion, optimizer, device, train=False, scaler=scaler, use_amp=use_amp
        )
        print(f"{'ETIS-LaribPolypDB':<20}{split_name:<10}{dice:<12.4f}{iou:<12.4f}{prec:<12.4f}{rec:<12.4f}{count:<8}")
    print("=" * 80)


    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved under: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)