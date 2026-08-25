"""
run_loo_experiment.py
Runs the full leave-one-dataset-out experiment: trains once per dataset in
config.yaml's `datasets` block, holding that dataset out each time and pooling
everything else for training (exactly what build_cross_dataset_splits()
already does for a single held_out_dataset - this just automates it across
every dataset instead of the one you'd set by hand).

Each fold gets its OWN checkpoint_dir/output_dir:
    <config's checkpoint_dir>/loo/<dataset name>
    <config's output_dir>/loo/<dataset name>
so folds never overwrite each other's checkpoints or results, and each fold's
`resume: true` checkpointing keeps working independently - if this script (or
your PC) dies mid-fold, just rerun it: completed folds are skipped instantly,
and the in-progress fold resumes via its own resume_checkpoint.pt exactly like
a normal single-fold run would.

Each fold runs as a SEPARATE subprocess (not an in-process function call) so
CUDA memory/context is fully reset between folds - matters more with SegFormer
than it did with SegNet, transformer models tend to leave more fragmented
CUDA cache behind.

Usage:
    python run_loo_experiment.py --config config.yaml
    python run_loo_experiment.py --config config.yaml --only CVC-ColonDB,Kvasir-SEG
    python run_loo_experiment.py --config config.yaml --force-rerun
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml


def main(cfg_path, only_datasets, force_rerun):
    with open(cfg_path) as f:
        base_cfg = yaml.safe_load(f)

    all_dataset_names = list(base_cfg["datasets"].keys())
    fold_names = only_datasets if only_datasets else all_dataset_names
    for name in fold_names:
        if name not in all_dataset_names:
            raise ValueError(f"'{name}' not found in config.yaml's datasets block. "
                              f"Available: {all_dataset_names}")

    base_ckpt_dir = Path(base_cfg["checkpoint_dir"])
    base_out_dir = Path(base_cfg["output_dir"])
    loo_configs_dir = Path("configs_loo")
    loo_configs_dir.mkdir(exist_ok=True)

    print(f"Leave-one-out experiment: {len(fold_names)} fold(s) -> {fold_names}\n")

    summary = {}

    for fold_name in fold_names:
        fold_ckpt_dir = base_ckpt_dir / "loo" / fold_name
        fold_out_dir = base_out_dir / "loo" / fold_name
        fold_test_results = fold_out_dir / "test_results.json"

        print("=" * 70)
        print(f"FOLD: held-out dataset = {fold_name}")
        print(f"  checkpoint_dir: {fold_ckpt_dir}")
        print(f"  output_dir:     {fold_out_dir}")
        print("=" * 70)

        if fold_test_results.exists() and not force_rerun:
            print(f"  Already completed (found {fold_test_results}), skipping. "
                  f"Use --force-rerun to redo it.\n")
            with open(fold_test_results) as f:
                summary[fold_name] = json.load(f)
            continue

        fold_cfg = copy.deepcopy(base_cfg)
        fold_cfg["held_out_dataset"] = fold_name
        fold_cfg["checkpoint_dir"] = str(fold_ckpt_dir)
        fold_cfg["output_dir"] = str(fold_out_dir)

        fold_cfg_path = loo_configs_dir / f"config_{fold_name}.yaml"
        with open(fold_cfg_path, "w") as f:
            yaml.safe_dump(fold_cfg, f, sort_keys=False)

        print(f"  Wrote fold config to {fold_cfg_path}, launching train.py ...\n")
        result = subprocess.run([sys.executable, "train.py", "--config", str(fold_cfg_path)])

        if result.returncode != 0:
            print(f"\n[ERROR] Training failed for fold '{fold_name}' (exit code {result.returncode}).")
            print("Fix the issue and rerun run_loo_experiment.py - completed folds are skipped, "
                  "and this fold picks up from its own resume_checkpoint.pt if it got partway through.")
            sys.exit(result.returncode)

        if fold_test_results.exists():
            with open(fold_test_results) as f:
                summary[fold_name] = json.load(f)
        else:
            print(f"[WARN] train.py exited cleanly but {fold_test_results} wasn't found - "
                  f"skipping this fold in the summary.")

    print_summary(summary)
    save_summary(summary, base_out_dir / "loo" / "loo_summary.json")


def print_summary(summary):
    if not summary:
        print("\nNo completed folds to summarize yet.")
        return

    import statistics as stats

    print("\n" + "=" * 70)
    print("LEAVE-ONE-OUT CROSS-DATASET SUMMARY")
    print("=" * 70)
    print(f"{'Held-out dataset':<22}{'Dice':<10}{'IoU':<10}{'Pixel Acc':<10}{'n_images':<10}")
    dices, ious, accs = [], [], []
    for name, r in summary.items():
        acc_val = r.get("test_acc", float("nan"))
        print(f"{name:<22}{r['test_dice']:<10.4f}{r['test_iou']:<10.4f}"
              f"{acc_val:<10.4f}{r['n_test_images']:<10}")
        dices.append(r["test_dice"]); ious.append(r["test_iou"])
        if "test_acc" in r:
            accs.append(r["test_acc"])

    print("-" * 70)
    if len(dices) > 1:
        print(f"{'MEAN':<22}{stats.mean(dices):<10.4f}{stats.mean(ious):<10.4f}"
              f"{(stats.mean(accs) if accs else float('nan')):<10.4f}")
        print(f"{'STD':<22}{stats.stdev(dices):<10.4f}{stats.stdev(ious):<10.4f}"
              f"{(stats.stdev(accs) if len(accs) > 1 else float('nan')):<10.4f}")
    print("=" * 70)
    print("\nA low mean Dice with a high std across folds is the same shortcut-learning /")
    print("domain-shift signature you saw in the YOLOv8 work - worth checking which specific")
    print("fold(s) are dragging the average down rather than treating it as one number.")


def save_summary(summary, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--only", type=str, default=None,
                         help="Comma-separated subset of dataset names to run as held-out folds "
                              "(default: every dataset in config.yaml)")
    parser.add_argument("--force-rerun", action="store_true",
                         help="Rerun folds even if test_results.json already exists for them")
    args = parser.parse_args()
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    main(args.config, only, args.force_rerun)