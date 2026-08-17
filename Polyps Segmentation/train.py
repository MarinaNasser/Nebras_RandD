"""
train.py
Trains SegNet (pretrained VGG-19 encoder) on 2 of the 3 polyp datasets and
evaluates on the held-out 3rd, per config.yaml.

Includes:
  - Automatic mixed precision (AMP): roughly halves per-kernel GPU time on
    most NVIDIA GPUs, which also helps avoid Windows' TDR watchdog killing
    long-running kernels (see README "Windows TDR" section).
  - Mid-epoch checkpointing + auto-resume: if training is interrupted
    (TDR crash, power loss, etc.), rerunning this script picks back up
    from the last saved step instead of starting over.

Usage:
    python train.py --config config.yaml
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
import matplotlib.pyplot as plt

from dataset import PolypDataset, build_cross_dataset_splits, get_train_transforms, get_val_transforms
from model import build_model
from metrics import BCEDiceLoss, dice_coeff, iou_score, pixel_accuracy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_resume_checkpoint(path, model, optimizer, scaler, epoch, step_in_epoch,
                            history, best_val_dice, encoder_unfrozen):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "history": history,
        "best_val_dice": best_val_dice,
        "encoder_unfrozen": encoder_unfrozen,
    }, path)


def run_epoch(model, loader, criterion, optimizer, device, train, scaler, use_amp,
              resume_ckpt_path=None, save_every_n_steps=0, epoch=0, history=None,
              best_val_dice=None, encoder_unfrozen=None):
    """
    If train=True and save_every_n_steps > 0, a resume checkpoint is written
    every N steps (in addition to the usual end-of-epoch checkpointing in
    main()), so an interrupted epoch doesn't lose all its progress.
    """
    model.train() if train else model.eval()
    total_loss, total_dice, total_iou, total_acc, n_batches = 0.0, 0.0, 0.0, 0.0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc="train" if train else "val", leave=False)
        for step, (images, masks) in enumerate(pbar, start=1):
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
            total_dice += dice_coeff(probs, masks).item()
            total_iou += iou_score(probs, masks).item()
            total_acc += pixel_accuracy(probs, masks).item()
            n_batches += 1
            pbar.set_postfix(loss=loss.item())

            if train and save_every_n_steps > 0 and step % save_every_n_steps == 0:
                save_resume_checkpoint(resume_ckpt_path, model, optimizer, scaler,
                                        epoch, step, history, best_val_dice, encoder_unfrozen)

    return total_loss / n_batches, total_dice / n_batches, total_iou / n_batches, total_acc / n_batches


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Defensive cast: guards against YAML quirks where scientific notation without
    # a decimal point (e.g. "1e-4" instead of "1.0e-4") gets parsed as a string
    # instead of a float, which would otherwise fail deep inside torch.optim.Adam
    # with a confusing "'<=' not supported between float and str" error.
    for key in ["lr", "lr_finetune", "weight_decay", "internal_val_fraction"]:
        cfg[key] = float(cfg[key])
    for key in ["image_size", "batch_size", "epochs", "num_workers", "seed",
                "early_stopping_patience", "freeze_encoder_epochs", "in_channels", "num_classes"]:
        cfg[key] = int(cfg[key])
    use_amp = bool(cfg.get("use_amp", True))
    save_every_n_steps = int(cfg.get("save_every_n_steps", 50))

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | AMP: {use_amp and device.type == 'cuda'}")

    ckpt_dir = Path(cfg["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    resume_ckpt_path = ckpt_dir / "resume_checkpoint.pt"

    # ---- Data ----
    train_pairs, internal_val_pairs, test_pairs = build_cross_dataset_splits(cfg)
    train_names_used = [name for name in cfg["datasets"] if name != cfg["held_out_dataset"]]
    image_size = cfg["image_size"]

    train_ds = PolypDataset(train_pairs, transform=get_train_transforms(image_size))
    internal_val_ds = PolypDataset(internal_val_pairs, transform=get_val_transforms(image_size))
    test_ds = PolypDataset(test_pairs, transform=get_val_transforms(image_size))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"], pin_memory=True)
    internal_val_loader = DataLoader(internal_val_ds, batch_size=cfg["batch_size"], shuffle=False,
                                      num_workers=cfg["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)

    # ---- Model (transfer learning: pretrained VGG-19 encoder, not trained from scratch) ----
    model = build_model(cfg.get("model_variant", "modified"), num_classes=cfg["num_classes"],
                         pretrained=True).to(device)

    criterion = BCEDiceLoss(bce_weight=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": [],
               "train_iou": [], "val_iou": [], "train_acc": [], "val_acc": []}
    best_val_dice = -1.0
    epochs_no_improve = 0
    start_epoch = 1
    encoder_unfrozen = False

    # ---- Resume from a mid-run crash (e.g. Windows TDR) if a checkpoint exists ----
    if resume_ckpt_path.exists() and cfg.get("resume", True):
        print(f"\n>>> Found resume checkpoint at {resume_ckpt_path}, resuming from there.")
        ckpt = torch.load(resume_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        history = ckpt["history"]
        for key in ("train_acc", "val_acc"):  # backward-compat if resuming an older checkpoint
            history.setdefault(key, [])
        best_val_dice = ckpt["best_val_dice"]
        encoder_unfrozen = ckpt["encoder_unfrozen"]
        start_epoch = ckpt["epoch"]  # re-run this epoch from scratch (simplest safe behavior);
                                      # partial in-epoch progress before the crash is not replayed,
                                      # but the model weights up to that point ARE kept.
        print(f"    resuming at epoch {start_epoch}, best_val_dice so far: {best_val_dice:.4f}, "
              f"encoder_unfrozen: {encoder_unfrozen}")

    model.set_encoder_trainable(encoder_unfrozen)
    if encoder_unfrozen:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr_finetune"], weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"],
                                      weight_decay=cfg["weight_decay"])
    if resume_ckpt_path.exists() and cfg.get("resume", True):
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception:
            print("    (optimizer state didn't match current optimizer, continuing with a fresh optimizer)")
        if ckpt["scaler_state"] is not None:
            scaler.load_state_dict(ckpt["scaler_state"])

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        # unfreeze encoder for fine-tuning after freeze_encoder_epochs
        if epoch == cfg["freeze_encoder_epochs"] + 1 and not encoder_unfrozen:
            print(f"\n>>> Unfreezing encoder for fine-tuning at epoch {epoch}")
            model.set_encoder_trainable(True)
            encoder_unfrozen = True
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr_finetune"],
                                          weight_decay=cfg["weight_decay"])

        train_loss, train_dice, train_iou, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True,
            scaler=scaler, use_amp=use_amp, resume_ckpt_path=resume_ckpt_path,
            save_every_n_steps=save_every_n_steps, epoch=epoch, history=history,
            best_val_dice=best_val_dice, encoder_unfrozen=encoder_unfrozen)
        val_loss, val_dice, val_iou, val_acc = run_epoch(
            model, internal_val_loader, criterion, optimizer, device, train=False,
            scaler=scaler, use_amp=use_amp)

        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice); history["val_dice"].append(val_dice)
        history["train_iou"].append(train_iou); history["val_iou"].append(val_iou)
        history["train_acc"].append(train_acc); history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"train_loss {train_loss:.4f} dice {train_dice:.4f} iou {train_iou:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} dice {val_dice:.4f} iou {val_iou:.4f} acc {val_acc:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            print(f"  -> new best internal val Dice {best_val_dice:.4f}, checkpoint saved")
        else:
            epochs_no_improve += 1

        # end-of-epoch resume checkpoint (covers the "epoch finished but crashed before the next one" case)
        save_resume_checkpoint(resume_ckpt_path, model, optimizer, scaler, epoch + 1, 0,
                                history, best_val_dice, encoder_unfrozen)

        if epochs_no_improve >= cfg["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch} (no improvement for "
                  f"{cfg['early_stopping_patience']} epochs)")
            break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ---- Plots ----
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, key, title in zip(axes, ["loss", "dice", "iou", "acc"], ["Loss", "Dice", "IoU", "Pixel Accuracy"]):
        ax.plot(history[f"train_{key}"], label="train")
        ax.plot(history[f"val_{key}"], label="internal val")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    print(f"Saved training curves to {out_dir / 'training_curves.png'}")

    # ---- Final evaluation on the held-out (never-seen) 3rd dataset ----
    print(f"\nLoading best checkpoint and evaluating on held-out dataset: {cfg['held_out_dataset']}")
    model.load_state_dict(torch.load(ckpt_dir / "best_model.pt", map_location=device))
    test_loss, test_dice, test_iou, test_acc = run_epoch(model, test_loader, criterion, optimizer, device,
                                                          train=False, scaler=scaler, use_amp=use_amp)

    result = {
        "held_out_dataset": cfg["held_out_dataset"],
        "test_loss": test_loss,
        "test_dice": test_dice,
        "test_iou": test_iou,
        "test_acc": test_acc,
        "n_test_images": len(test_pairs),
    }
    with open(out_dir / "test_results.json", "w") as f:
        json.dump(result, f, indent=2)

    # ---- Clean final summary (also re-printable anytime via show_results.py) ----
    best_epoch = int(np.argmax(history["val_dice"])) + 1
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"Model variant:        {cfg.get('model_variant', 'modified')}")
    print(f"Trained on:            {', '.join(train_names_used)}")
    print(f"Held out (test) set:   {cfg['held_out_dataset']}  ({len(test_pairs)} images)")
    print(f"Epochs run:            {len(history['train_loss'])} / {cfg['epochs']} (best at epoch {best_epoch})")
    print("-" * 60)
    print(f"{'Metric':<12}{'Train (last epoch)':<22}{'Internal Val (best)':<22}")
    print(f"{'Loss':<12}{history['train_loss'][-1]:<22.4f}{history['val_loss'][best_epoch-1]:<22.4f}")
    print(f"{'Dice':<12}{history['train_dice'][-1]:<22.4f}{history['val_dice'][best_epoch-1]:<22.4f}")
    print(f"{'IoU':<12}{history['train_iou'][-1]:<22.4f}{history['val_iou'][best_epoch-1]:<22.4f}")
    print(f"{'Pixel Acc':<12}{history['train_acc'][-1]:<22.4f}{history['val_acc'][best_epoch-1]:<22.4f}")
    print("-" * 60)
    print(f"HELD-OUT TEST SET ({cfg['held_out_dataset']}):")
    print(f"  Loss: {test_loss:.4f}   Dice: {test_dice:.4f}   IoU: {test_iou:.4f}   Pixel Acc: {test_acc:.4f}")
    print("=" * 60)
    print(f"\nFull history saved to:      {out_dir / 'history.json'}")
    print(f"Test results saved to:      {out_dir / 'test_results.json'}")
    print(f"Training curves saved to:   {out_dir / 'training_curves.png'}")
    print(f"\nRe-print this summary anytime with: python show_results.py --config {cfg_path}")

    # training finished cleanly - the resume checkpoint is no longer needed
    if resume_ckpt_path.exists():
        resume_ckpt_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
