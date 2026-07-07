"""
rerun_eval_all_wavelet_checkpoints.py
============================================================
Loads every saved wavelet_gan_generator_epoch*.pt checkpoint and
re-runs the full module6 evaluation suite on the SAME test split used
during training. Imports get_dataloaders() directly from module7_train.py
(not reimplemented) to guarantee an identical patch-level test split
(same manifest, same seed=42) — critical for valid comparison against
the epoch100 numbers already obtained and against GAN/U-Net baselines.

Does NOT retrain. Only reloads checkpoints + re-evaluates.

Usage:
    python rerun_eval_all_wavelet_checkpoints.py --manifest E:\SAR-Optical-Synthesis\data\manifest.csv
"""

import argparse
import glob
import os
import re

import torch

from module7_train import get_dataloaders, set_seed
from module14_wavelet_gan import WaveletAttentionUNet
from module6_evaluation import evaluate_batch, summarize


def evaluate_checkpoint(ckpt_path, test_loader, device):
    generator = WaveletAttentionUNet(in_channels=9, out_channels=6, base_ch=64).to(device)
    generator.load_state_dict(torch.load(ckpt_path, map_location=device))
    generator.eval()

    all_results = None
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            preds = generator(inputs)

            batch_results = evaluate_batch(preds, targets)
            if all_results is None:
                all_results = {k: [] for k in batch_results}
            for k in all_results:
                all_results[k].extend(batch_results[k])

    print(f"\n{'='*70}\nCheckpoint: {os.path.basename(ckpt_path)}\n{'='*70}")
    summary = summarize(all_results)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CRITICAL: same seed + same get_dataloaders() call as module7_train.py,
    # so this test split is bit-for-bit identical to the one used to
    # produce your reported epoch100 numbers.
    set_seed(42)
    train_loader, val_loader, test_loader = get_dataloaders(args.manifest, batch_size=args.batch_size)

    ckpt_paths = sorted(
        glob.glob(os.path.join(args.checkpoint_dir, "wavelet_gan_generator_epoch*.pt")),
        key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)),
    )

    if not ckpt_paths:
        print(f"No wavelet_gan_generator_epoch*.pt checkpoints found in {args.checkpoint_dir}")
        return

    print(f"Found {len(ckpt_paths)} checkpoints: {[os.path.basename(p) for p in ckpt_paths]}")

    all_summaries = {}
    for ckpt_path in ckpt_paths:
        epoch_num = int(re.search(r"epoch(\d+)", ckpt_path).group(1))
        summary = evaluate_checkpoint(ckpt_path, test_loader, device)
        all_summaries[epoch_num] = summary

    print(f"\n{'='*70}\nCROSS-CHECKPOINT COMPARISON (key metrics)\n{'='*70}")
    header = (f"{'Epoch':>6} | {'PSNR':>7} | {'SSIM':>6} | {'NDVI_R2':>8} | "
              f"{'NDWI_R2':>8} | {'MNDWI_R2':>9} | {'BSI_R2':>8} | {'Shoreline_m':>11}")
    print(header)
    print("-" * len(header))
    for epoch_num in sorted(all_summaries.keys()):
        s = all_summaries[epoch_num]
        print(f"{epoch_num:>6} | "
              f"{s['image_psnr']['mean']:>7.3f} | "
              f"{s['image_ssim']['mean']:>6.3f} | "
              f"{s['ndvi_r2']['mean']:>8.4f} | "
              f"{s['ndwi_r2']['mean']:>8.4f} | "
              f"{s['mndwi_r2']['mean']:>9.4f} | "
              f"{s['bsi_r2']['mean']:>8.4f} | "
              f"{s['ndwi_shoreline_error_m']['mean']:>11.2f}")


if __name__ == "__main__":
    main()