"""
MODULE 8 — BSI Diagnostic
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
Investigates why BSI R^2 is weak (GAN: 0.13, UNet: -0.13) compared to
NDVI/NDWI/MNDWI (GAN: 0.73-0.80) on the test set.

Two checks, run on the EXACT same seed=42 patch-level test split used
for the reported results (get_dataloaders() in module7_train.py):

1. PER-BAND reconstruction quality (RMSE + PSNR for each of the 6
   output bands individually, not averaged together like module6 does).
   BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
   so if B11 (SWIR1) is reconstructed worse than B2/B3/B4/B8, that's
   a likely direct cause.

2. GROUND-TRUTH BSI variance/distribution in the test set. Low R^2 can
   happen even with good reconstruction if the ground-truth values
   themselves have low variance (e.g. mostly water/vegetation, little
   bare soil) — R^2 punishes low-variance targets harshly regardless of
   model quality. This tells us if it's a DATA issue, not a MODEL issue.

Does NOT retrain anything. Loads the saved GAN checkpoint
(checkpoints/gan_generator_epoch100.pt) and runs inference once on the
test split.

Usage:
    python module8_bsi_diagnostic.py --manifest E:\\SAR-Optical-Synthesis\\data\\manifest.csv --checkpoint E:\\SAR-Optical-Synthesis\\checkpoints\\gan_generator_epoch100.pt
"""

import argparse
import numpy as np
import torch

from module2_dataset import SAROpticalTripletDataset, OPTICAL_BAND_ORDER
from module4_attention_unet import AttentionUNet
from module7_train import get_dataloaders  # reuses identical seed=42 split logic

BAND_NAMES = OPTICAL_BAND_ORDER  # ["B2","B3","B4","B8","B11","B12"]


def per_band_metrics(preds, targets):
    """preds, targets: (N, 6, H, W) numpy arrays in [0,1]."""
    results = {}
    for i, name in enumerate(BAND_NAMES):
        p = preds[:, i, :, :]
        g = targets[:, i, :, :]
        rmse = float(np.sqrt(np.mean((p - g) ** 2)))
        mae = float(np.mean(np.abs(p - g)))
        # per-pixel R^2 across this band globally
        g_flat = g.flatten()
        p_flat = p.flatten()
        ss_res = np.sum((g_flat - p_flat) ** 2)
        ss_tot = np.sum((g_flat - g_flat.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        results[name] = {"rmse": rmse, "mae": mae, "r2": r2, "gt_std": float(g_flat.std())}
    return results


def compute_bsi_np(optical):
    """optical: (..., 6, H, W) -> BSI map. Matches module6_evaluation.compute_bsi."""
    EPS = 1e-6
    swir1 = optical[..., 4, :, :]  # B11
    red = optical[..., 2, :, :]    # B4
    nir = optical[..., 3, :, :]    # B8
    blue = optical[..., 0, :, :]   # B2
    return ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + EPS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # IMPORTANT: same seed=42 split as module7_train.py -> identical test set
    _, _, test_loader = get_dataloaders(args.manifest, batch_size=args.batch_size)

    model = AttentionUNet(in_channels=9, out_channels=6, base_ch=64).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            preds = model(inputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)   # (N, 6, H, W)
    targets = np.concatenate(all_targets, axis=0)
    print(f"\nTest set size: {preds.shape[0]} samples\n")

    # --- Check 1: per-band reconstruction quality ---
    print("=" * 70)
    print("CHECK 1 — Per-band reconstruction quality")
    print("=" * 70)
    band_results = per_band_metrics(preds, targets)
    print(f"{'Band':6s} {'RMSE':>8s} {'MAE':>8s} {'R2':>8s} {'GT_std':>8s}")
    for name, m in band_results.items():
        print(f"{name:6s} {m['rmse']:8.4f} {m['mae']:8.4f} {m['r2']:8.4f} {m['gt_std']:8.4f}")
    print("\nBSI depends on: B11 (SWIR1), B4 (Red), B8 (NIR), B2 (Blue).")
    print("If B11 has a notably higher RMSE / lower R2 / lower GT_std than")
    print("B2/B3/B4/B8, that's a likely direct contributor to weak BSI.")

    # --- Check 2: ground-truth BSI variance ---
    print("\n" + "=" * 70)
    print("CHECK 2 — Ground-truth BSI distribution in test set")
    print("=" * 70)
    gt_bsi = compute_bsi_np(targets)        # (N, H, W)
    pred_bsi = compute_bsi_np(preds)

    print(f"GT BSI   -> mean={gt_bsi.mean():.4f}  std={gt_bsi.std():.4f}  "
          f"min={gt_bsi.min():.4f}  max={gt_bsi.max():.4f}")
    print(f"Pred BSI -> mean={pred_bsi.mean():.4f}  std={pred_bsi.std():.4f}  "
          f"min={pred_bsi.min():.4f}  max={pred_bsi.max():.4f}")

    # crude bare-soil proxy: fraction of pixels with BSI > 0 (commonly used
    # as a rough bare-soil/built-up threshold in remote sensing practice —
    # not a calibrated classification, just a sanity signal here)
    frac_bare_gt = float((gt_bsi > 0).mean())
    print(f"\nFraction of GT pixels with BSI > 0 (rough bare-soil/built-up proxy): "
          f"{frac_bare_gt:.4f}")
    print("If this fraction is very low (e.g. <5%), the test set may simply")
    print("contain little bare-soil signal -> low variance -> R2 is punished")
    print("structurally, independent of model quality. If it's not low, the")
    print("weak R2 is more likely a genuine reconstruction-quality issue tied")
    print("to Check 1's per-band results.")

    # per-sample BSI std, to see if variance is low broadly or just a few
    # outlier patches are dragging things down
    per_sample_std = gt_bsi.reshape(gt_bsi.shape[0], -1).std(axis=1)
    print(f"\nPer-sample GT BSI std -> mean={per_sample_std.mean():.4f}, "
          f"median={np.median(per_sample_std):.4f}, "
          f"min={per_sample_std.min():.4f}, max={per_sample_std.max():.4f}")


if __name__ == "__main__":
    main()