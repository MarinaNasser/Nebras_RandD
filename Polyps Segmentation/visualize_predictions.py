"""
visualize_predictions.py
Loads the best checkpoint and saves a grid of (image, ground truth, prediction)
for a handful of images from the held-out test dataset - useful as a qualitative
sanity check alongside the Dice/IoU numbers in test_results.json.

Usage:
    python visualize_predictions.py --config config.yaml --n 8
"""
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import build_cross_dataset_splits, PolypDataset, get_val_transforms
from model import build_model

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def denormalize(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * STD + MEAN).clip(0, 1)
    return img


def main(cfg_path, n):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_pairs = build_cross_dataset_splits(cfg)
    test_ds = PolypDataset(test_pairs, transform=get_val_transforms(cfg["image_size"]))

    model = build_model(cfg.get("model_variant", "modified"), num_classes=cfg["num_classes"],
                         pretrained=False, segformer_size=cfg.get("segformer_size", "b0")).to(device)
    model.load_state_dict(torch.load(f"{cfg['checkpoint_dir']}/best_model.pt", map_location=device))
    model.eval()

    idxs = np.random.choice(len(test_ds), size=min(n, len(test_ds)), replace=False)
    fig, axes = plt.subplots(len(idxs), 3, figsize=(9, 3 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for row, idx in enumerate(idxs):
            image, mask = test_ds[idx]
            logits = model(image.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits) > 0.5).float().cpu()[0, 0]

            axes[row, 0].imshow(denormalize(image)); axes[row, 0].set_title("Image"); axes[row, 0].axis("off")
            axes[row, 1].imshow(mask[0], cmap="gray"); axes[row, 1].set_title("Ground truth"); axes[row, 1].axis("off")
            axes[row, 2].imshow(pred, cmap="gray"); axes[row, 2].set_title("Prediction"); axes[row, 2].axis("off")

    plt.tight_layout()
    out_path = f"{cfg['output_dir']}/qualitative_predictions.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved qualitative predictions to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()
    main(args.config, args.n)
