"""
MODULE 2 — PyTorch Dataset and DataLoader
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
Loads triplets built by module1_build_manifest.py.

Input tensor  (9 channels, 256x256):
    0: VV (SAR)
    1: VH (SAR)
    2-7: B2,B3,B4,B8,B11,B12 — cloud-free optical REFERENCE (pre-gap date)
    8: cloud_mask of the reference scene (0=clear,1=was cloudy there)
       NOTE: this mask describes the REFERENCE image's own cloud cover,
       not a synthetic blackout. It tells the model "the reference is
       unreliable in these pixels, weight SAR more heavily here" — this
       is genuine auxiliary information, unlike the rejected idea of
       reusing it as a loss-weighting mask (see module3_loss.py header
       for why that was rejected).

Target tensor (6 channels, 256x256):
    B2,B3,B4,B8,B11,B12 — REAL, held-out, genuinely-different-date
    cloud-free optical ground truth.

PATCH_SIZE = 256, locked. GEE patch-grid script already exports
patches pre-sized to 256x256, so no further cropping/sliding-window
extraction happens here — if your downloaded files are NOT already
256x256, fix the GEE patch grid (gee_scripts/01_aoi_and_patch_grid.js),
don't silently resize here (resizing would corrupt SAR backscatter and
reflectance values, which carry physical meaning).

Only rows with has_gt=True are loaded by default (require_gt=True) —
this is what feeds your reported quantitative metrics. If you want a
larger training set including no-GT rows for self-supervised/qualitative
use, set require_gt=False, but keep that in a SEPARATE dataloader from
your test-set evaluation, never mixed.
"""

import csv
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

PATCH_SIZE = 256          # LOCKED — see module docstring
INPUT_CHANNELS = 9
TARGET_CHANNELS = 6
OPTICAL_BAND_ORDER = ["B2", "B3", "B4", "B8", "B11", "B12"]
INPUT_BAND_NAMES = ["VV", "VH"] + OPTICAL_BAND_ORDER + ["CLOUD_MASK_REF"]


def read_tif_as_array(path, expected_bands=None):
    with rasterio.open(path) as src:
        arr = src.read()  # (bands, H, W)
        if expected_bands is not None and arr.shape[0] != expected_bands:
            raise ValueError(
                f"Band count mismatch in {path}: expected {expected_bands}, got {arr.shape[0]}. "
                f"Check the GEE export band order before proceeding — do not silently continue."
            )
        if arr.shape[1] != PATCH_SIZE or arr.shape[2] != PATCH_SIZE:
            raise ValueError(
                f"Patch size mismatch in {path}: expected {PATCH_SIZE}x{PATCH_SIZE}, "
                f"got {arr.shape[1]}x{arr.shape[2]}. Fix the GEE patch grid script, "
                f"do not resize here."
            )
        return arr.astype(np.float32)


def normalize_sar(db_values):
    """Sentinel-1 GRD VV/VH (COPERNICUS/S1_GRD) are already in dB.
    Clip to a sane dB range and rescale to [0,1].
    TODO: confirm this matches the normalization actually applied (or not
    applied) in the GEE export script — mismatched normalization between
    GEE export and Python is a classic silent bug."""
    clipped = np.clip(db_values, -25, 0)
    return (clipped + 25) / 25.0


def normalize_optical(reflectance):
    """Sentinel-2 SR reflectance is typically scaled by 10000 in GEE.
    TODO: confirm whether the GEE export script already divided by 10000.
    If yes, do NOT divide again here. If no, this is the right place."""
    return np.clip(reflectance / 10000.0, 0.0, 1.0)


class SAROpticalTripletDataset(Dataset):
    """
    Each item:
        input:  torch.Tensor (9, 256, 256) float32
        target: torch.Tensor (6, 256, 256) float32

    Augmentation (train split only):
        - random horizontal flip
        - random vertical flip
        - random 90/180/270 rotation
      Applied identically to input AND target (they are spatial pairs).
      No spectral/color augmentation — SAR backscatter and optical
      reflectance carry physical meaning that must not be altered.
    """

    def __init__(self, manifest_csv, require_gt=True, augment=False):
        self.rows = []
        with open(manifest_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                has_gt = row["has_gt"] in ("True", "true", "1")
                if require_gt and not has_gt:
                    continue
                self.rows.append(row)

        self.require_gt = require_gt
        self.augment = augment

        if len(self.rows) == 0:
            raise RuntimeError(
                "No usable rows found in manifest with require_gt="
                f"{require_gt}. Check module1_build_manifest.py's printed "
                "counts before debugging this class further."
            )

        print(f"SAROpticalTripletDataset: {len(self.rows)} samples "
              f"(require_gt={require_gt}, augment={augment})")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        sar = read_tif_as_array(row["sar_path"], expected_bands=2)         # VV, VH
        ref = read_tif_as_array(row["ref_path"], expected_bands=7)         # 6 optical + cloud mask
        ref_optical = ref[:6]
        ref_cloud_mask = ref[6:7]

        sar_norm = normalize_sar(np.nan_to_num(sar, nan=0.0))
        ref_norm = normalize_optical(ref_optical)
        # ref_cloud_mask already 0/1, no normalization needed

        input_stack = np.concatenate([sar_norm, ref_norm, ref_cloud_mask], axis=0)  # (9,256,256)

        if row["has_gt"] in ("True", "true", "1") and row["gt_path"]:
            gt = read_tif_as_array(row["gt_path"], expected_bands=6)
            target = normalize_optical(gt)
        else:
            # placeholder — only reached if require_gt=False; never used in
            # quantitative loss/eval as long as that flag is respected upstream
            target = np.zeros_like(ref_norm)

        input_tensor = torch.from_numpy(input_stack)
        target_tensor = torch.from_numpy(target)

        if self.augment:
            input_tensor, target_tensor = self._augment(input_tensor, target_tensor)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "patch_id": row["patch_id"],
            "aoi": row.get("aoi", "unknown"),
            "window_start": row["window_start"],
            "sar_date": row["sar_date"],
        }

    def _augment(self, inp, tgt):
        if random.random() > 0.5:
            inp, tgt = TF.hflip(inp), TF.hflip(tgt)
        if random.random() > 0.5:
            inp, tgt = TF.vflip(inp), TF.vflip(tgt)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            inp, tgt = TF.rotate(inp, angle), TF.rotate(tgt, angle)
        return inp, tgt


# ============================================================
# QUICK VERIFICATION — run standalone to sanity check a manifest
# ============================================================
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    import os

    if len(sys.argv) < 2:
        print("Usage: python module2_dataset.py /path/to/manifest.csv")
        sys.exit(1)

    manifest_path = sys.argv[1]
    ds = SAROpticalTripletDataset(manifest_path, require_gt=True, augment=False)

    sample = ds[0]
    inp, tgt = sample["input"], sample["target"]

    print(f"\nSample 0:")
    print(f"  patch_id: {sample['patch_id']}  aoi: {sample['aoi']}")
    print(f"  input shape:  {inp.shape}  range [{inp.min():.3f}, {inp.max():.3f}]")
    print(f"  target shape: {tgt.shape}  range [{tgt.min():.3f}, {tgt.max():.3f}]")

    assert not torch.isnan(inp).any(), "NaN in input!"
    assert not torch.isnan(tgt).any(), "NaN in target!"
    assert not torch.isinf(inp).any(), "Inf in input!"
    assert not torch.isinf(tgt).any(), "Inf in target!"
    print("  No NaN/Inf detected ✓")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    inp_np, tgt_np = inp.numpy(), tgt.numpy()

    axes[0].imshow(inp_np[0], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("SAR VV (input)")
    axes[0].axis("off")

    rgb_ref = np.clip(np.stack([inp_np[4], inp_np[3], inp_np[2]], axis=-1) * 3.5, 0, 1)
    axes[1].imshow(rgb_ref)
    axes[1].set_title("Reference optical RGB (input, pre-gap)")
    axes[1].axis("off")

    axes[2].imshow(inp_np[8], cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[2].set_title("Reference cloud mask")
    axes[2].axis("off")

    rgb_gt = np.clip(np.stack([tgt_np[2], tgt_np[1], tgt_np[0]], axis=-1) * 3.5, 0, 1)
    axes[3].imshow(rgb_gt)
    axes[3].set_title("Ground-truth optical RGB (target, real gap-date)")
    axes[3].axis("off")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(manifest_path), "sample_triplet_check.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved verification figure: {out_path}")
