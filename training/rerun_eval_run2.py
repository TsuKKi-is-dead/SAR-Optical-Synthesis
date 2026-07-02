"""
RERUN EVALUATION — Run 2 Unweighted GAN + U-Net (ablation)
============================================================
Purpose: Re-derive the Table 1 metrics from the saved checkpoint
so the paper has a verified CSV source, not just the handoff prompt.

Uses IDENTICAL split logic to module7_train.py:
  - seed=42, val_split=0.15, test_split=0.15, patch-level split
  - band_weights=None (unweighted — the original reported run)

Usage (run from E:\\SAR-Optical-Synthesis with venv active):
    python rerun_eval_run2.py ^
        --manifest E:\\SAR-Optical-Synthesis\\data\\manifest.csv ^
        --gan_checkpoint E:\\SAR-Optical-Synthesis\\checkpoints_run2_unweighted\\gan_generator_epoch100.pt ^
        --unet_checkpoint E:\\SAR-Optical-Synthesis\\checkpoints_run2_unweighted\\unet_best.pt ^
        --output_csv E:\\SAR-Optical-Synthesis\\training\\eval_results_run2_verified.csv

If you don't have a saved U-Net checkpoint from Run 2, omit
--unet_checkpoint and only GAN metrics will be written to the CSV.
The GAN numbers are what matter for Table 1's primary column.
"""

import argparse
import csv
import os
import numpy as np
import torch

# These imports assume you run from E:\SAR-Optical-Synthesis
# (i.e. the directory containing module*.py files)
from module4_attention_unet import AttentionUNet
from module6_evaluation import evaluate_batch, summarize
from module7_train import get_dataloaders


def load_generator(checkpoint_path, device):
    model = AttentionUNet(in_channels=9, out_channels=6, base_ch=64).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded: {checkpoint_path}")
    return model


def run_evaluation(model, test_loader, device, model_name):
    all_results = None
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            preds = model(inputs)
            batch_results = evaluate_batch(preds, targets)
            if all_results is None:
                all_results = {k: [] for k in batch_results}
            for k in all_results:
                all_results[k].extend(batch_results[k])

    print(f"\n=== {model_name} — Test Set Results ===")
    summary = summarize(all_results)
    return summary


def flatten_summary(summary, model_name):
    """Convert summary dict to flat rows for CSV output."""
    rows = []
    for metric, stats in summary.items():
        rows.append({
            "model": model_name,
            "metric": metric,
            "mean": round(stats["mean"], 4) if not np.isnan(stats["mean"]) else "nan",
            "std": round(stats["std"], 4) if not np.isnan(stats["std"]) else "nan",
            "n": stats["n"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True,
                        help="Path to manifest.csv (module1 output)")
    parser.add_argument("--gan_checkpoint", required=True,
                        help="Path to gan_generator_epoch100.pt")
    parser.add_argument("--unet_checkpoint", default=None,
                        help="Path to unet_best.pt (optional)")
    parser.add_argument("--output_csv", default="eval_results_run2_verified.csv",
                        help="Where to save verified metric table")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # IDENTICAL split to module7_train.py — seed=42, patch-level
    print("\nBuilding test split (seed=42, identical to training run)...")
    _, _, test_loader = get_dataloaders(
        args.manifest,
        batch_size=args.batch_size,
        val_split=0.15,
        test_split=0.15,
        seed=42,
    )

    all_rows = []

    # GAN (primary)
    print("\nLoading GAN generator...")
    gan = load_generator(args.gan_checkpoint, device)
    gan_summary = run_evaluation(gan, test_loader, device, "GAN (primary)")
    all_rows.extend(flatten_summary(gan_summary, "GAN"))

    # U-Net (ablation) — optional
    if args.unet_checkpoint and os.path.exists(args.unet_checkpoint):
        print("\nLoading U-Net...")
        unet = load_generator(args.unet_checkpoint, device)
        unet_summary = run_evaluation(unet, test_loader, device, "AttentionUNet (ablation)")
        all_rows.extend(flatten_summary(unet_summary, "UNet"))
    else:
        print("\nNo U-Net checkpoint provided or found — skipping U-Net evaluation.")
        print("GAN metrics only will be written to CSV.")

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "metric", "mean", "std", "n"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'='*60}")
    print(f"Verified metrics saved to: {args.output_csv}")
    print(f"{'='*60}")
    print("\nPaper Table 1 key values to check against handoff prompt:")
    print(f"  {'Metric':<30} {'Handoff value':<20} {'This run'}")
    print(f"  {'-'*70}")

    targets = {
        "GAN": {
            "image_psnr": 33.30,
            "image_ssim": 0.873,
            "ndvi_r2": 0.734,
            "ndwi_r2": 0.776,
            "mndwi_r2": 0.797,
            "bsi_r2": 0.132,
            "ndwi_shoreline_error_m": 189.0,
        },
        "UNet": {
            "image_psnr": 29.17,
            "image_ssim": 0.818,
            "ndvi_r2": -0.328,
            "ndwi_r2": 0.075,
            "mndwi_r2": 0.032,
            "bsi_r2": -0.132,
            "ndwi_shoreline_error_m": 174.0,
        },
    }

    for row in all_rows:
        model = row["model"]
        metric = row["metric"]
        handoff = targets.get(model, {}).get(metric, "—")
        match = ""
        if handoff != "—" and row["mean"] != "nan":
            diff = abs(float(row["mean"]) - float(handoff))
            match = "✓" if diff < 0.5 else f"⚠ diff={diff:.3f}"
        print(f"  {model+'/'+metric:<30} {str(handoff):<20} {row['mean']}  {match}")

    print("\nIf all key metrics show ✓, Table 1 is verified. Upload the CSV and")
    print("I will remove the warning comment from main.tex.")


if __name__ == "__main__":
    main()