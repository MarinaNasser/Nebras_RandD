"""
show_results.py
Prints the training/validation history and held-out test metrics from a
completed (or in-progress) run, reading straight from the saved JSON files -
no retraining or GPU needed.

Usage:
    python show_results.py --config config.yaml
    python show_results.py --config config.yaml --plot   # also (re)draws training_curves.png
"""
import argparse
import json
from pathlib import Path

import yaml


def main(cfg_path, replot):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output_dir"])
    history_path = out_dir / "history.json"
    test_path = out_dir / "test_results.json"

    if not history_path.exists():
        print(f"No {history_path} found yet - run train.py first (or wait for it to finish at least one epoch).")
        return

    with open(history_path) as f:
        history = json.load(f)

    n_epochs = len(history["train_loss"])
    best_epoch = max(range(n_epochs), key=lambda i: history["val_dice"][i]) + 1

    print("=" * 60)
    print("TRAINING HISTORY")
    print("=" * 60)
    print(f"Config:               {cfg_path}")
    print(f"Model variant:        {cfg.get('model_variant', 'modified')}")
    print(f"Held-out (test) set:  {cfg['held_out_dataset']}")
    print(f"Epochs completed:     {n_epochs} / {cfg['epochs']}")
    print(f"Best epoch (by internal val Dice): {best_epoch}")
    print("-" * 60)
    print(f"{'Epoch':<8}{'Train Loss':<13}{'Train Dice':<13}{'Val Loss':<13}{'Val Dice':<13}")
    for i in range(n_epochs):
        marker = "  <- best" if (i + 1) == best_epoch else ""
        print(f"{i+1:<8}{history['train_loss'][i]:<13.4f}{history['train_dice'][i]:<13.4f}"
              f"{history['val_loss'][i]:<13.4f}{history['val_dice'][i]:<13.4f}{marker}")

    print("-" * 60)
    print(f"Best internal val Dice: {history['val_dice'][best_epoch-1]:.4f}  "
          f"(IoU: {history['val_iou'][best_epoch-1]:.4f}"
          + (f", Pixel Acc: {history['val_acc'][best_epoch-1]:.4f})" if "val_acc" in history else ")"))

    if test_path.exists():
        with open(test_path) as f:
            result = json.load(f)
        print("\n" + "=" * 60)
        print(f"HELD-OUT TEST SET RESULTS: {result['held_out_dataset']} "
              f"({result['n_test_images']} images)")
        print("=" * 60)
        print(f"  Loss: {result['test_loss']:.4f}")
        print(f"  Dice: {result['test_dice']:.4f}")
        print(f"  IoU:  {result['test_iou']:.4f}")
        if "test_acc" in result:
            print(f"  Pixel Acc: {result['test_acc']:.4f}")
    else:
        print(f"\n(No {test_path} yet - final held-out evaluation runs after all training epochs finish.)")

    if replot:
        import matplotlib.pyplot as plt
        keys = ["loss", "dice", "iou"] + (["acc"] if "train_acc" in history else [])
        titles = ["Loss", "Dice", "IoU"] + (["Pixel Accuracy"] if "train_acc" in history else [])
        fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
        if len(keys) == 1:
            axes = [axes]
        for ax, key, title in zip(axes, keys, titles):
            ax.plot(history[f"train_{key}"], label="train")
            ax.plot(history[f"val_{key}"], label="internal val")
            ax.axvline(best_epoch - 1, color="gray", linestyle="--", alpha=0.5, label="best epoch")
            ax.set_title(title); ax.set_xlabel("epoch"); ax.legend()
        plt.tight_layout()
        out_path = out_dir / "training_curves.png"
        plt.savefig(out_path, dpi=150)
        print(f"\nRe-saved training curves to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--plot", action="store_true", help="Also (re)generate training_curves.png")
    args = parser.parse_args()
    main(args.config, args.plot)
