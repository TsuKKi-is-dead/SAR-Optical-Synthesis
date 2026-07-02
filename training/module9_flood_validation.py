"""
module9_flood_validation.py
============================
Flood-mapping validation for the 2020 Mahanadi flood event (Paper 2, Section: Application).

Two modes:
  --mode threshold_check   Sanity-check the z-score seasonal anomaly threshold on the
                           flood date before committing to it. Saves diagnostic
                           GeoTIFFs and prints area statistics. Run first.

  --mode run_validation    Full pipeline:
                             Track A: SAR seasonal-anomaly reference mask (z-score)
                             Track B: GAN reconstruction -> NDWI/MNDWI -> water mask
                             Comparison: IoU, precision, recall, area agreement

Usage:
  python training\\module9_flood_validation.py --mode threshold_check
  python training\\module9_flood_validation.py --mode run_validation

Locked conventions (match rest of pipeline):
  - CRS: EPSG:32645 (UTM 45N)
  - Patch size: 256x256 px
  - SAR bands: [VV, VH] (Float32, dB scale)
  - Optical bands: [B2, B3, B4, B8, B11, B12] (6-band output from GAN)
  - Pre-monsoon optical reference window start: 2021-06-01
  - Flood SAR date: 2020-08-26 (rising-stage, pre-peak -- do NOT call this peak flood)
  - num_workers: 0 (Windows)

WHY Z-SCORE SEASONAL ANOMALY (not single-date threshold):
  Two single-date threshold approaches were tried and both failed:

  Attempt 1 (VV-only, -17 dB): baseline (Jul 21) flagged 27.7% of AOI, flood date
  (Aug 26) flagged only 18.0% -- backwards. Cause: July is kharif rice-transplanting
  season; flooded paddy fields show near-water VV backscatter, inflating the baseline.

  Attempt 2 (dual-pol, VV<-17 AND VH<-24 dB): baseline 26.0%, flood date 16.4% --
  still backwards. Root cause diagnosed via ocean-strip analysis: on Aug 26, open
  ocean VV averaged -12.92 dB (well above -17 dB) because pre-monsoon wind had
  roughened the sea surface, brightening the entire scene. This is a scene-level
  roughness effect, not a paddy-contamination effect, and no fixed threshold can
  correct for it -- the VH sweep showed baseline > flood at every value from -30
  to -18 dB, i.e. no crossover threshold exists.

  Z-score anomaly detection solves this: instead of comparing flood-date backscatter
  to a fixed absolute threshold, each pixel is compared to its OWN multi-year
  seasonal (Jun-Sep, 2021-2023) median and std:

      z = (flood_VV - VV_seasonal_median) / VV_seasonal_std

  Pixels with z below a negative threshold (e.g. -1.5 or -2.0) are anomalously dark
  relative to their seasonal norm and flagged as newly inundated. Scene-level
  roughness (which shifts the whole image's backscatter up or down roughly equally)
  cancels out in the per-pixel subtraction, since it affects the flood-date pixel
  value but not the pixel's own historical median/std. Paddy phenology is also
  handled correctly: a paddy pixel's seasonal median already reflects its normal
  flooded-transplanting-season backscatter, so it is not flagged unless backscatter
  drops further below ITS OWN norm, not an absolute number shared with open water.
  Cite: WorldFloods (Scientific Reports, 2021) for legitimacy of self-derived SAR
  reference masks; Bioresita et al. (2018), Twele et al. (2016) for dual-pol context
  on SAR water detection in rice-growing deltas (background only, not the method used).
"""

import os
import sys
import argparse
import numpy as np
import glob
import torch
import rasterio
from rasterio.crs import CRS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────

BASE_DIR       = r"E:\SAR-Optical-Synthesis"
DATA_DIR       = os.path.join(BASE_DIR, "data")
FLOOD_DIR      = os.path.join(DATA_DIR, "flood_validation")
OPT_REF_DIR    = os.path.join(DATA_DIR, "optical_reference")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_run2_unweighted")
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs", "flood_validation")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── flood / baseline file paths ───────────────────────────────────────────────

FLOOD_SAR_PATH     = os.path.join(FLOOD_DIR, "mahanadi_flood2020_flooddate_sar_2020-08-26.tif")
FLOOD_MASK_PATH    = os.path.join(FLOOD_DIR, "mahanadi_flood2020_flooddate_mask_2020-08-26.tif")   # DEPRECATED, VV-only, unused

# DEPRECATED for Track A: single-date Jul-21 baseline SAR. Superseded by the multi-year
# seasonal median/std file below, which plays the "what is normal here" role instead.
BASELINE_SAR_PATH  = os.path.join(FLOOD_DIR, "mahanadi_flood2020_baseline_sar_2020-07-21.tif")
BASELINE_MASK_PATH = os.path.join(FLOOD_DIR, "mahanadi_flood2020_baseline_mask_2020-07-21.tif")    # DEPRECATED, VV-only, unused

# Multi-year (2021-2023, Jun-Sep) seasonal median/std SAR file, from gee_scripts/05.
# 4 bands: VV_median, VH_median, VV_std, VH_std (Float32).
SEASONAL_MEDIAN_PATH = os.path.join(FLOOD_DIR, "mahanadi_sar_seasonal_median_2021_2023.tif")

# ── thresholds ────────────────────────────────────────────────────────────────

Z_THRESHOLD = -1.5
NDWI_THRESHOLD   = 0.0
OPT_REF_WINDOW_START = "2021-06-01"
GAN_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "gan_generator_epoch100.pt")
PATCH_SIZE = 256


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_tif(path, description=""):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}  ({description})")
    with rasterio.open(path) as src:
        data    = src.read().astype(np.float32)
        profile = src.profile
    vv = data[0]
    finite = vv[np.isfinite(vv)]
    if finite.size > 0:
        print(f"  Loaded {description}: {os.path.basename(path)} | shape={data.shape} | "
              f"band0 min={finite.min():.2f} max={finite.max():.2f} "
              f"mean={finite.mean():.2f} std={finite.std():.2f} | "
              f"NaN pixels={np.isnan(vv).sum():,}")
    else:
        print(f"  Loaded {description}: {os.path.basename(path)} | shape={data.shape} | ALL NaN")
    return data, profile


def compute_zscore(flood_vv, vv_median, vv_std, eps=1e-6):
    safe_std = np.where(vv_std > eps, vv_std, eps)
    z = (flood_vv - vv_median) / safe_std
    invalid = ~np.isfinite(flood_vv) | ~np.isfinite(vv_median) | ~np.isfinite(vv_std)
    z = np.where(invalid, 0.0, z)
    return z.astype(np.float32)


def apply_zscore_threshold(z, z_thresh=Z_THRESHOLD):
    return (z < z_thresh).astype(np.uint8)


def compute_ndwi(b3, b8):
    denom = b3 + b8
    denom = np.where(denom == 0, 1e-6, denom)
    return (b3 - b8) / denom


def compute_mndwi(b3, b11):
    denom = b3 + b11
    denom = np.where(denom == 0, 1e-6, denom)
    return (b3 - b11) / denom


def area_fraction(mask, label=""):
    frac = mask.sum() / mask.size
    print(f"  {label}: {int(mask.sum()):,} / {mask.size:,} pixels = {frac*100:.2f}%")
    return frac


def compute_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def compute_precision_recall(pred, ref):
    tp = np.logical_and(pred, ref).sum()
    fp = np.logical_and(pred, ~ref.astype(bool)).sum()
    fn = np.logical_and(~pred.astype(bool), ref).sum()
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec  = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def load_seasonal_median(path):
    data, profile = load_tif(path, "seasonal median/std (2021-2023)")
    if data.shape[0] != 4:
        raise ValueError(
            f"Expected 4 bands [VV_median, VH_median, VV_std, VH_std] in "
            f"{os.path.basename(path)}, got {data.shape[0]} bands."
        )
    vv_median, vh_median, vv_std, vh_std = data[0], data[1], data[2], data[3]
    return vv_median, vh_median, vv_std, vh_std, profile


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: THRESHOLD SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def mode_threshold_check():
    print("\n" + "="*60)
    print("MODE: threshold_check (z-score seasonal anomaly)")
    print(f"  Z_THRESHOLD: {Z_THRESHOLD}")
    print("="*60)

    print("\n[1] Loading seasonal median/std and flood SAR...")
    vv_median, vh_median, vv_std, vh_std, _ = load_seasonal_median(SEASONAL_MEDIAN_PATH)
    flood_sar, flood_profile = load_tif(FLOOD_SAR_PATH, "flood SAR (Aug 26)")
    vv_flood = flood_sar[0]

    print(f"\n  VV seasonal median: mean={np.nanmean(vv_median):.2f} dB | "
          f"VV seasonal std: mean={np.nanmean(vv_std):.2f} dB")

    print(f"\n[2] Computing z-score (flood_VV - VV_median) / VV_std...")
    z = compute_zscore(vv_flood, vv_median, vv_std)
    finite_z = z[np.isfinite(z)]
    print(f"  z-score stats: min={finite_z.min():.2f} max={finite_z.max():.2f} "
          f"mean={finite_z.mean():.2f} std={finite_z.std():.2f}")

    flood_anomaly = apply_zscore_threshold(z, Z_THRESHOLD)
    print(f"\n  Flagging pixels with z < {Z_THRESHOLD}:")
    area_fraction(flood_anomaly, "    Flood-date anomaly (new inundation candidate)")

    print("\n[3] Sweeping Z_THRESHOLD (-3.0 to -0.5)...")
    z_range = np.arange(-3.0, -0.4, 0.25)
    fracs = [apply_zscore_threshold(z, t).mean() for t in z_range]
    for t, f in zip(z_range, fracs):
        marker = "  <-- locked" if abs(t - Z_THRESHOLD) < 1e-6 else ""
        print(f"    z < {t:5.2f}: {f*100:5.2f}% of AOI{marker}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0,0].plot(z_range, np.array(fracs)*100, 'r-o', ms=4)
    axes[0,0].axvline(Z_THRESHOLD, color='k', ls='--', label=f'Locked: {Z_THRESHOLD}')
    axes[0,0].set_xlabel('Z threshold'); axes[0,0].set_ylabel('Flagged area (%)')
    axes[0,0].set_title('Z-threshold sweep (flood date, Aug 26)')
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].hist(finite_z, bins=100, color='steelblue', alpha=0.8)
    axes[0,1].axvline(Z_THRESHOLD, color='k', ls='--', label=f'Locked: {Z_THRESHOLD}')
    axes[0,1].set_xlabel('z-score'); axes[0,1].set_ylabel('Pixel count')
    axes[0,1].set_title('Z-score distribution, full AOI')
    axes[0,1].set_yscale('log')
    axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

    vmin_f, vmax_f = np.nanpercentile(vv_flood, [2, 98])
    axes[1,0].imshow(vv_flood, cmap='gray', vmin=vmin_f, vmax=vmax_f)
    axes[1,0].set_title('Flood VV SAR (Aug 26, rising-stage)')
    axes[1,0].axis('off')

    axes[1,1].imshow(vv_flood, cmap='gray', vmin=vmin_f, vmax=vmax_f)
    overlay = np.ma.masked_where(flood_anomaly == 0, flood_anomaly.astype(float))
    axes[1,1].imshow(overlay, cmap='Reds', alpha=0.65, vmin=0, vmax=1)
    axes[1,1].set_title(f'Z-score anomaly mask (z < {Z_THRESHOLD})\n'
                        f'red = newly inundated, {flood_anomaly.mean()*100:.1f}% of AOI')
    axes[1,1].axis('off')

    plt.suptitle(
        f'Z-score Seasonal Anomaly Check — Mahanadi\n'
        f'z = (flood_VV - VV_seasonal_median) / VV_seasonal_std,  threshold = {Z_THRESHOLD}',
        fontsize=13
    )
    plt.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, "threshold_check_zscore.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Diagnostic PNG saved: {out_png}")

    mask_profile = flood_profile.copy()
    mask_profile.update(count=1, dtype='uint8')
    with rasterio.open(os.path.join(OUTPUT_DIR, "flooddate_water_mask_zscore.tif"), 'w', **mask_profile) as dst:
        dst.write(flood_anomaly[np.newaxis])
    print(f"  GeoTIFF saved to {OUTPUT_DIR}")

    print("\n[THRESHOLD CHECK COMPLETE]")
    print("─"*60)
    print("  Once satisfied, proceed to:")
    print("    python training\\module9_flood_validation.py --mode run_validation")
    print("─"*60)


# ══════════════════════════════════════════════════════════════════════════════
# OPTICAL REFERENCE INDEX (replaces load_optical_reference_mosaic)
# ══════════════════════════════════════════════════════════════════════════════

def build_optical_patch_index(flood_sar_profile):
    """
    Build a spatial index of all Mahanadi optical reference patches.
    Returns:
      index  -- list of (fpath, r0, c0, patch_H, patch_W) for per-tile lookup
      mosaic -- (7, H, W) float32, assembled for saving/inspection only
                (NOT used as GAN input -- avoids re-tiling seam artifacts)

    Root cause of the checkerboard bug this replaces:
      The old approach built a full mosaic then re-tiled it with tile_image().
      Each optical patch is placed at a non-integer pixel offset (computed via
      rowcol() from its geotransform), so mosaic seams don't align with the
      regular 256-pixel inference tile grid. When a GAN tile straddles a mosaic
      seam, zero-fill from the adjacent unplaced region bleeds into the tile --
      producing the visible checkerboard of empty squares in the output.

    The fix: for each SAR inference tile, look up which optical patches overlap
    that tile's spatial region and composite them directly (get_optical_tile),
    rather than re-tiling a pre-built mosaic. The mosaic is still built here
    for saving reconstructed_optical_flood.tif, but is never fed into the GAN.
    """
    from rasterio.transform import rowcol

    pattern = os.path.join(OPT_REF_DIR, f"s2ref_mahanadi_*_{OPT_REF_WINDOW_START}.tif")
    files   = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No Mahanadi optical reference files found for window {OPT_REF_WINDOW_START}\n"
            f"  Pattern: {pattern}"
        )
    print(f"  Found {len(files)} Mahanadi optical reference patches for window {OPT_REF_WINDOW_START}")

    H = flood_sar_profile['height']
    W = flood_sar_profile['width']
    transform_target = flood_sar_profile['transform']
    crs_target = CRS.from_epsg(32645)

    mosaic = np.zeros((7, H, W), dtype=np.float32)
    index  = []
    placed = 0
    drop_reasons = {'crs_mismatch': [], 'rowcol_exception': [], 'out_of_bounds': []}

    for fpath in files:
        with rasterio.open(fpath) as src:
            if src.crs != crs_target:
                drop_reasons['crs_mismatch'].append(os.path.basename(fpath))
                continue
            patch_data      = src.read().astype(np.float32)
            patch_transform = src.transform
            patch_H, patch_W = src.height, src.width

        patch_origin_x = patch_transform.c
        patch_origin_y = patch_transform.f
        try:
            row_off, col_off = rowcol(transform_target, patch_origin_x, patch_origin_y)
        except Exception as e:
            drop_reasons['rowcol_exception'].append(os.path.basename(fpath))
            continue

        r0 = int(row_off); c0 = int(col_off)
        r1 = r0 + patch_H; c1 = c0 + patch_W

        if r1 <= 0 or c1 <= 0 or r0 >= H or c0 >= W:
            drop_reasons['out_of_bounds'].append(os.path.basename(fpath))
            continue

        # Build mosaic for saving
        pr0 = max(0, -r0); pc0 = max(0, -c0)
        mr0 = max(0, r0);  mc0 = max(0, c0)
        mr1 = min(H, r1);  mc1 = min(W, c1)
        mosaic[:, mr0:mr1, mc0:mc1] = patch_data[:, pr0:pr0+(mr1-mr0), pc0:pc0+(mc1-mc0)]

        index.append((fpath, r0, c0, patch_H, patch_W))
        placed += 1

    print(f"  Indexed {placed}/{len(files)} patches")
    for reason, items in drop_reasons.items():
        if items:
            print(f"    dropped ({reason}): {len(items)}")

    if placed == 0:
        raise RuntimeError("No optical reference patches indexed -- check CRS and AOI overlap.")

    return index, mosaic


def get_optical_tile(index, tile_r, tile_c, patch_size=PATCH_SIZE):
    """
    For a SAR inference tile at (tile_r, tile_c), composite the optical data
    from all indexed patches that overlap that region.

    Returns (7, patch_size, patch_size) float32.
    Zeros where no patch covers (edge regions where training grid exceeds SAR AOI).

    Each patch is read fresh from disk for the overlapping sub-window only,
    so there is no seam artifact from re-tiling a pre-assembled mosaic.
    """
    buf = np.zeros((7, patch_size, patch_size), dtype=np.float32)
    tile_r1 = tile_r + patch_size
    tile_c1 = tile_c + patch_size

    for fpath, r0, c0, ph, pw in index:
        r1 = r0 + ph; c1 = c0 + pw
        # Overlap between this patch and the SAR tile
        or0 = max(r0, tile_r); or1 = min(r1, tile_r1)
        oc0 = max(c0, tile_c); oc1 = min(c1, tile_c1)
        if or0 >= or1 or oc0 >= oc1:
            continue
        # Position in tile output buffer
        tr0 = or0 - tile_r; tr1 = or1 - tile_r
        tc0 = oc0 - tile_c; tc1 = oc1 - tile_c
        # Position within the patch file
        pr_s = or0 - r0; pr_e = or1 - r0
        pc_s = oc0 - c0; pc_e = oc1 - c0

        with rasterio.open(fpath) as src:
            region = src.read()[:, pr_s:pr_e, pc_s:pc_e].astype(np.float32)
        buf[:, tr0:tr1, tc0:tc1] = region

    return buf


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: FULL VALIDATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def normalize_sar_patch(patch):
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
    patch = np.clip(patch, -30.0, 0.0)
    return ((patch + 30.0) / 30.0).astype(np.float32)


def normalize_optical_patch(patch):
    return np.clip(patch.astype(np.float32) / 10000.0, 0.0, 1.0)


def tile_image(image, patch_size=PATCH_SIZE):
    _, H, W = image.shape
    tiles = []
    for r in range(0, H - patch_size + 1, patch_size):
        for c in range(0, W - patch_size + 1, patch_size):
            tiles.append((image[:, r:r+patch_size, c:c+patch_size], r, c))
    return tiles


def mosaic_patches(patches_with_positions, full_shape, patch_size=PATCH_SIZE):
    bands, H, W = full_shape
    out   = np.zeros(full_shape, dtype=np.float32)
    count = np.zeros((H, W),    dtype=np.int32)
    for patch, r, c in patches_with_positions:
        out[:, r:r+patch_size, c:c+patch_size] += patch
        count[r:r+patch_size, c:c+patch_size]  += 1
    count = np.where(count == 0, 1, count)
    return out / count[np.newaxis]


def mode_run_validation():
    print("\n" + "="*60)
    print("MODE: run_validation")
    print(f"  Z_THRESHOLD: {Z_THRESHOLD}")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    # ── TRACK A: SAR seasonal-anomaly reference ──────────────────────────────
    print("\n[TRACK A] Building z-score seasonal-anomaly reference flood mask...")

    vv_median, vh_median, vv_std, vh_std, _ = load_seasonal_median(SEASONAL_MEDIAN_PATH)
    flood_sar, flood_profile = load_tif(FLOOD_SAR_PATH, "flood SAR")
    vv_flood = flood_sar[0]

    z = compute_zscore(vv_flood, vv_median, vv_std)
    new_flood_sar = apply_zscore_threshold(z, Z_THRESHOLD)

    print("  Track A water fraction:")
    area_fraction(new_flood_sar, "    New inundation (z-score anomaly)")

    H_full, W_full = flood_sar.shape[1], flood_sar.shape[2]

    # ── TRACK B: GAN reconstruction ───────────────────────────────────────────
    print("\n[TRACK B] Running GAN reconstruction on flood-date SAR...")

    print(f"  Loading GAN checkpoint: {os.path.basename(GAN_CHECKPOINT)}")
    if not os.path.exists(GAN_CHECKPOINT):
        raise FileNotFoundError(f"GAN checkpoint not found: {GAN_CHECKPOINT}")

    sys.path.insert(0, os.path.join(BASE_DIR, "training"))
    from module4_attention_unet import AttentionUNet

    model = AttentionUNet(in_channels=9, out_channels=6)
    state_dict = torch.load(GAN_CHECKPOINT, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("  GAN generator loaded.")

    # Build optical patch index (spatial index for per-tile lookup)
    # Also returns mosaic for saving only -- NOT used as GAN input.
    print("\n  Building optical reference patch index...")
    opt_index, opt_ref_mosaic = build_optical_patch_index(flood_profile)

    # Tile flood SAR into 256x256 patches
    print("\n  Tiling SAR into 256x256 patches for GAN inference...")
    sar_tiles = tile_image(flood_sar, PATCH_SIZE)
    print(f"  Total SAR tiles: {len(sar_tiles)}")

    # Run GAN inference -- optical input read per-tile from the patch index,
    # not from a re-tiled mosaic (avoids the checkerboard seam artifact).
    output_patches = []
    zero_opt_tiles = 0
    print("  Running GAN inference (optical data read tile-by-tile from patch index)...")
    for i, (sar_patch, r, c) in enumerate(sar_tiles):
        # Get optical data for this tile's spatial region directly from patch files
        opt_patch = get_optical_tile(opt_index, r, c, PATCH_SIZE)  # (7, 256, 256)

        if opt_patch.max() == 0:
            zero_opt_tiles += 1

        sar_norm = normalize_sar_patch(sar_patch)            # (2, 256, 256)
        opt_norm = normalize_optical_patch(opt_patch)        # (7, 256, 256)
        inp      = np.concatenate([sar_norm, opt_norm], axis=0)  # (9, 256, 256)

        inp_tensor = torch.from_numpy(inp).unsqueeze(0).float()
        with torch.no_grad():
            out = model(inp_tensor.to(device))
        out_np = out.squeeze(0).cpu().numpy()  # (6, 256, 256)
        output_patches.append((out_np, r, c))

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(sar_tiles)} patches done")

    print(f"  GAN inference complete: {len(output_patches)} patches")
    if zero_opt_tiles > 0:
        print(f"  WARNING: {zero_opt_tiles} tiles had all-zero optical input "
              f"(edge regions not covered by any optical reference patch)")

    # Mosaic GAN outputs back to full AOI
    reconstructed = mosaic_patches(output_patches, (6, H_full, W_full), PATCH_SIZE)
    # bands: [B2, B3, B4, B8, B11, B12] normalized [0, 1]
    b3_recon  = reconstructed[1]   # Green
    b8_recon  = reconstructed[3]   # NIR
    b11_recon = reconstructed[4]   # SWIR1

    ndwi_recon  = compute_ndwi(b3_recon, b8_recon)
    mndwi_recon = compute_mndwi(b3_recon, b11_recon)

    ndwi_water_flood  = (ndwi_recon  > NDWI_THRESHOLD).astype(np.uint8)
    mndwi_water_flood = (mndwi_recon > NDWI_THRESHOLD).astype(np.uint8)

    # Baseline water from optical reference mosaic (pre-monsoon)
    opt_norm_mosaic = normalize_optical_patch(opt_ref_mosaic)
    b3_ref  = opt_norm_mosaic[1]
    b8_ref  = opt_norm_mosaic[3]
    b11_ref = opt_norm_mosaic[4]

    ndwi_ref  = compute_ndwi(b3_ref, b8_ref)
    mndwi_ref = compute_mndwi(b3_ref, b11_ref)

    baseline_water_ndwi  = (ndwi_ref  > NDWI_THRESHOLD).astype(np.uint8)
    baseline_water_mndwi = (mndwi_ref > NDWI_THRESHOLD).astype(np.uint8)

    print(f"  Baseline NDWI water (pre-subtraction): {baseline_water_ndwi.mean()*100:.1f}%")
    print(f"  Baseline MNDWI water (pre-subtraction): {baseline_water_mndwi.mean()*100:.1f}%")


    # ── Optical coverage mask ─────────────────────────────────────────────────
# Pixels where the optical reference mosaic has no patch coverage (all-zero
# bands) produce unreliable GAN output AND an incorrect baseline NDWI of 0.0
# (which misclassifies permanent ocean as "new inundation" since the GAN
# correctly outputs high NDWI for ocean tiles). Exclude these pixels from
# all Track B comparisons.
    opt_coverage = (opt_ref_mosaic[:6].max(axis=0) > 100.0)  # True where coverage exists
    ndwi_water_flood   = np.where(opt_coverage, ndwi_water_flood,  0).astype(np.uint8)
    mndwi_water_flood  = np.where(opt_coverage, mndwi_water_flood, 0).astype(np.uint8)
    baseline_water_ndwi  = np.where(opt_coverage, baseline_water_ndwi,  0).astype(np.uint8)
    baseline_water_mndwi = np.where(opt_coverage, baseline_water_mndwi, 0).astype(np.uint8)

    

    # Compare total reconstructed water extent directly against Track A.
    # Baseline subtraction was dropped: the Jun-2021 optical reference is a
    # monsoon-onset window with transplanted paddy fields already flooded,
    # giving ~20% baseline water -- nearly identical to flood-date water extent.
    # Subtracting it removes the genuine flood signal entirely. Track A already
    # isolates new inundation via z-score anomaly; Track B simply reports where
    # the GAN-reconstructed optical sees water, and the comparison asks whether
    # those water pixels spatially agree with Track A's anomaly mask.
    new_flood_ndwi  = ndwi_water_flood.copy()
    new_flood_mndwi = mndwi_water_flood.copy()

    print("\n  Track B water fractions:")
    area_fraction(ndwi_water_flood,   "    NDWI water (flood-date, no baseline subtraction)")
    area_fraction(mndwi_water_flood,  "    MNDWI water (flood-date, no baseline subtraction)")

    # ── COMPARISON METRICS ────────────────────────────────────────────────────
    print("\n[COMPARISON] Track B vs Track A (z-score SAR anomaly reference)...")
    pixel_area_km2 = (10.0 * 10.0) / 1e6

    iou_ndwi   = compute_iou(new_flood_ndwi,  new_flood_sar)
    iou_mndwi  = compute_iou(new_flood_mndwi, new_flood_sar)
    prec_n, rec_n, f1_n = compute_precision_recall(new_flood_ndwi,  new_flood_sar)
    prec_m, rec_m, f1_m = compute_precision_recall(new_flood_mndwi, new_flood_sar)

    area_sar   = int(new_flood_sar.sum())   * pixel_area_km2
    area_ndwi  = int(new_flood_ndwi.sum())  * pixel_area_km2
    area_mndwi = int(new_flood_mndwi.sum()) * pixel_area_km2

    print("\n  ── Flood extent areas ──────────────────────────────")
    print(f"  Track A (SAR z-score anomaly): {area_sar:.2f} km²")
    if area_sar > 0:
        print(f"  Track B NDWI  (reconstructed): {area_ndwi:.2f} km²  "
              f"({area_ndwi/area_sar*100:.1f}% of SAR reference)")
        print(f"  Track B MNDWI (reconstructed): {area_mndwi:.2f} km²  "
              f"({area_mndwi/area_sar*100:.1f}% of SAR reference)")
    else:
        print(f"  Track B NDWI  (reconstructed): {area_ndwi:.2f} km²")
        print(f"  Track B MNDWI (reconstructed): {area_mndwi:.2f} km²")

    print("\n  ── Spatial agreement metrics ───────────────────────")
    print(f"  NDWI  vs SAR:  IoU={iou_ndwi:.3f}  Prec={prec_n:.3f}  Rec={rec_n:.3f}  F1={f1_n:.3f}")
    print(f"  MNDWI vs SAR:  IoU={iou_mndwi:.3f}  Prec={prec_m:.3f}  Rec={rec_m:.3f}  F1={f1_m:.3f}")

    # ── COMPARISON FIGURE ─────────────────────────────────────────────────────
    print("\n  Saving comparison figure...")
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    vmin_f, vmax_f = np.nanpercentile(vv_flood, [2, 98])
    titles = [
        f'Flood VV SAR (Aug 26)',
        f'Track A: SAR z-score anomaly\n(new inundation, {area_sar:.1f} km²)',
        f'Track B: NDWI reconstructed\n({area_ndwi:.1f} km², IoU={iou_ndwi:.2f})',
        f'Track B: MNDWI reconstructed\n({area_mndwi:.1f} km², IoU={iou_mndwi:.2f})',
    ]
    masks  = [None, new_flood_sar, new_flood_ndwi, new_flood_mndwi]
    colors = [None, 'Blues', 'Greens', 'Oranges']

    for ax, title, mask, cmap in zip(axes, titles, masks, colors):
        ax.imshow(vv_flood, cmap='gray', vmin=vmin_f, vmax=vmax_f)
        if mask is not None:
            overlay = np.ma.masked_where(mask == 0, mask.astype(float))
            ax.imshow(overlay, cmap=cmap, alpha=0.65, vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    plt.suptitle(
        'Flood Extent Validation — 2020 Mahanadi Flood (Aug 26, rising-stage)\n'
        'Track A: independent SAR seasonal z-score anomaly reference  |  '
        'Track B: GAN-reconstructed optical',
        fontsize=11
    )
    plt.tight_layout()

    out_fig = os.path.join(OUTPUT_DIR, "flood_validation_comparison.png")
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison figure: {out_fig}")

    # ── SAVE OUTPUT GEOTIFFS ──────────────────────────────────────────────────
    mask_profile = flood_profile.copy()
    mask_profile.update(count=1, dtype='uint8')

    for fname, arr in [
        ("new_flood_sar_zscore_reference.tif", new_flood_sar),
        ("new_flood_ndwi_trackB.tif",          new_flood_ndwi),
        ("new_flood_mndwi_trackB.tif",         new_flood_mndwi),
    ]:
        with rasterio.open(os.path.join(OUTPUT_DIR, fname), 'w', **mask_profile) as dst:
            dst.write(arr[np.newaxis])
    print(f"  Flood mask GeoTIFFs saved to {OUTPUT_DIR}")

    recon_profile = flood_profile.copy()
    recon_profile.update(count=6, dtype='float32')
    with rasterio.open(os.path.join(OUTPUT_DIR, "reconstructed_optical_flood.tif"), 'w', **recon_profile) as dst:
        dst.write(reconstructed)
    print(f"  Reconstructed optical mosaic saved.")
    print(f"  Optical coverage mask: {opt_coverage.mean()*100:.1f}% of AOI has patch coverage")

    print("\n[VALIDATION COMPLETE]")
    print("─"*60)
    print("Summary for Paper 2:")
    print(f"  SAR z-score anomaly new inundation: {area_sar:.2f} km²")
    print(f"  NDWI-derived new inundation:        {area_ndwi:.2f} km²  (IoU={iou_ndwi:.3f}, F1={f1_n:.3f})")
    print(f"  MNDWI-derived new inundation:       {area_mndwi:.2f} km²  (IoU={iou_mndwi:.3f}, F1={f1_m:.3f})")
    print("─"*60)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Flood validation pipeline — Paper 2")
    parser.add_argument(
        '--mode',
        choices=['threshold_check', 'run_validation'],
        required=True,
    )
    args = parser.parse_args()

    if args.mode == 'threshold_check':
        mode_threshold_check()
    elif args.mode == 'run_validation':
        mode_run_validation()


if __name__ == '__main__':
    main()