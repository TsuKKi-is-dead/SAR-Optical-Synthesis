"""
MODULE 7 — Main Training Script
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
Trains the GAN (PRIMARY, as of the 100-epoch comparison — empirically
beat the deterministic U-Net on index-level R^2 across NDVI/NDWI/MNDWI)
and the Attention U-Net (ablation/baseline), on IDENTICAL patch-level
splits, then evaluates BOTH on the held-out test set with the full
metric battery (module6).

NEW: --band_weights flag, testing the BSI diagnostic experiment (see
module3_loss.py / module5_gan_baseline.py docstrings). Default is
unweighted (None), reproducing the exact original reported results.

Patch-level split (not scene-level): splits by patch_id so the same
physical 256x256 patch never appears in both train and val/test.

Usage:
    python module7_train.py --manifest /path/to/manifest.csv --model both --epochs 100
    python module7_train.py --manifest /path/to/manifest.csv --model gan --epochs 100 --band_weights 2.0,1.0,2.0,1.0,1.0,1.0
"""

import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from module2_dataset import SAROpticalTripletDataset
from module3_loss import CompositeLoss
from module4_attention_unet import AttentionUNet
from module5_gan_baseline import PatchDiscriminator, gan_training_step
from module6_evaluation import evaluate_batch, summarize

from module14_wavelet_gan import WaveletAttentionUNet, wavelet_gan_training_step
from module13_wavelet_utils import EdgeExtractor
from module5_gan_baseline import PatchDiscriminator

def train_wavelet_gan(train_loader, val_loader, epochs, device, lr=2e-4, band_weights=None):
    generator = WaveletAttentionUNet(in_channels=9, out_channels=6, base_ch=64).to(device)
    discriminator = PatchDiscriminator(in_channels=9 + 6).to(device)
    edge_extractor = EdgeExtractor(channels=6).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        generator.train(); discriminator.train()
        running = {"loss_d": 0.0, "loss_g_adv": 0.0, "loss_g_l1": 0.0, "loss_g_edge": 0.0}
        for batch in train_loader:
            stats = wavelet_gan_training_step(generator, discriminator, edge_extractor,
                                               opt_g, opt_d, batch, device=device, band_weights=band_weights)
            for k in running: running[k] += stats[k]
        n = max(len(train_loader), 1)
        print(f"[WaveletGAN] Epoch {epoch+1}/{epochs} " + " ".join(f"{k}={v/n:.4f}" for k,v in running.items()))
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f"checkpoints/wavelet_gan_generator_epoch{epoch+1}.pt")
    return generator


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # NOTE: also set torch.backends.cudnn.deterministic=True if you need
    # bit-exact reproducibility for the paper's reported numbers — this
    # can slow training, so decide consciously.


def make_subset(full_dataset, rows, augment):
    ds = SAROpticalTripletDataset.__new__(SAROpticalTripletDataset)
    ds.rows = rows
    ds.require_gt = True
    ds.augment = augment
    return ds


def get_dataloaders(manifest_csv, batch_size=8, val_split=0.15, test_split=0.15, seed=42):
    """
    IMPORTANT: split by patch_id, not by row. A single patch_id can have
    multiple manifest rows (different SAR dates within the same gap
    window) — all rows for a given patch_id MUST land in the same split,
    or the model could see a patch's reference scene in train and its
    paired SAR acquisition in test, leaking spatial information.
    """
    full_dataset = SAROpticalTripletDataset(manifest_csv, require_gt=True, augment=False)

    patch_ids = sorted(set(r["patch_id"] for r in full_dataset.rows))
    n = len(patch_ids)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    train_ids = set(patch_ids[i] for i in perm[:n_train])
    val_ids = set(patch_ids[i] for i in perm[n_train:n_train + n_val])
    test_ids = set(patch_ids[i] for i in perm[n_train + n_val:])

    train_rows = [r for r in full_dataset.rows if r["patch_id"] in train_ids]
    val_rows = [r for r in full_dataset.rows if r["patch_id"] in val_ids]
    test_rows = [r for r in full_dataset.rows if r["patch_id"] in test_ids]

    print(f"Patch-level split -> train: {len(train_ids)} patches / {len(train_rows)} samples, "
          f"val: {len(val_ids)} / {len(val_rows)}, test: {len(test_ids)} / {len(test_rows)}")

    if min(len(train_rows), len(val_rows), len(test_rows)) == 0:
        raise RuntimeError(
            "One of train/val/test has zero samples — your dataset is too "
            "small for this split ratio, or has_gt rows are too sparse. "
            "Check module1_build_manifest.py's printed counts."
        )

    train_ds = make_subset(full_dataset, train_rows, augment=True)
    val_ds = make_subset(full_dataset, val_rows, augment=False)
    test_ds = make_subset(full_dataset, test_rows, augment=False)

    nw = 0

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw, drop_last=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nw),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=nw),
    )


def train_unet(train_loader, val_loader, epochs, device, lr=1e-4, band_weights=None):
    model = AttentionUNet(in_channels=9, out_channels=6, base_ch=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = CompositeLoss(band_weights=band_weights).to(device)

    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            preds = model(inputs)
            loss, _ = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                preds = model(inputs)
                loss, _ = criterion(preds, targets)
                val_loss += loss.item()

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)
        print(f"[UNet] Epoch {epoch+1}/{epochs}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "checkpoints/unet_best.pt")

    return model


def train_gan(train_loader, val_loader, epochs, device, lr=2e-4, band_weights=None):
    generator = AttentionUNet(in_channels=9, out_channels=6, base_ch=64).to(device)
    discriminator = PatchDiscriminator(in_channels=9 + 6).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        running = {"loss_d": 0.0, "loss_g_adv": 0.0, "loss_g_l1": 0.0}
        for batch in train_loader:
            stats = gan_training_step(generator, discriminator, opt_g, opt_d, batch,
                                       device=device, band_weights=band_weights)
            for k in running:
                running[k] += stats[k]

        n = max(len(train_loader), 1)
        print(f"[GAN] Epoch {epoch+1}/{epochs}  "
              f"loss_d={running['loss_d']/n:.4f}  "
              f"loss_g_adv={running['loss_g_adv']/n:.4f}  "
              f"loss_g_l1={running['loss_g_l1']/n:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f"checkpoints/gan_generator_epoch{epoch+1}.pt")

    return generator


def evaluate_model(model, test_loader, device, model_name):
    model.eval()
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
    return summarize(all_results)


def parse_band_weights(s):
    if s is None:
        return None
    try:
        vals = [float(x) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "band_weights must be 6 comma-separated numbers, e.g. 2.0,1.0,2.0,1.0,1.0,1.0"
        )
    if len(vals) != 6:
        raise argparse.ArgumentTypeError(
            f"band_weights must have exactly 6 values (B2,B3,B4,B8,B11,B12), got {len(vals)}"
        )
    return vals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet", "gan", "wavelet_gan", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--manifest", type=str, required=True,
                         help="Path to manifest.csv produced by module1_build_manifest.py")
    parser.add_argument("--band_weights", type=str, default=None,
                         help="Optional 6 comma-separated weights for B2,B3,B4,B8,B11,B12 "
                              "L1 loss term (e.g. '2.0,1.0,2.0,1.0,1.0,1.0' to up-weight "
                              "B2/B4 per the BSI diagnostic). Default: unweighted, identical "
                              "to previously reported results.")
    args = parser.parse_args()

    band_weights = parse_band_weights(args.band_weights)
    if band_weights is not None:
        print(f"Using band_weights={band_weights} (B2,B3,B4,B8,B11,B12) — "
              f"NOT the original unweighted configuration.")

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(args.manifest, batch_size=args.batch_size)

    results_table = {}

    if args.model in ("unet", "both"):
        unet_model = train_unet(train_loader, val_loader, args.epochs, device, band_weights=band_weights)
        results_table["AttentionUNet"] = evaluate_model(unet_model, test_loader, device, "Attention U-Net")

    if args.model in ("gan", "both"):
        gan_generator = train_gan(train_loader, val_loader, args.epochs, device, band_weights=band_weights)
        results_table["GAN"] = evaluate_model(gan_generator, test_loader, device, "Pix2Pix-style GAN")

    if args.model in ("wavelet_gan", "both"):
        wavelet_generator = train_wavelet_gan(train_loader, val_loader, args.epochs, device, band_weights=band_weights)
        results_table["WaveletGAN"] = evaluate_model(wavelet_generator, test_loader, device, "Wavelet GAN (Li et al.-inspired)")

    print("\n=== FINAL COMPARISON TABLE (use this in the paper) ===")
    for model_name, metrics in results_table.items():
        print(f"\n{model_name}:")
        for metric_name, stats in metrics.items():
            print(f"  {metric_name}: {stats['mean']:.4f} +/- {stats['std']:.4f}  (n={stats['n']})")