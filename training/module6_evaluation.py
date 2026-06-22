"""
MODULE 6 — Evaluation Suite
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
Reports ALL FOUR metric families on the held-out test set:
  - R^2            (correlation/trend fidelity — matches your already-
                     submitted Ganjam erosion paper's metric convention,
                     so Paper 2 results are directly comparable to Paper 1)
  - RMSE           (absolute index-unit error — what actually matters for
                     DSAS/shoreline threshold-based detection)
  - SSIM-on-index  (spatial-pattern fidelity of the index map itself, not
                     just the raw image — catches "right values, wrong
                     place" errors; this is also what lets you cite/beat
                     CloudBreaker's reported NDWI/NDVI SSIM numbers
                     0.6156 / 0.6874 directly)
  - Shoreline error (metres) (your most reviewer-legible number, directly
                     comparable to Paper 1's reported shoreline accuracy)

Why all four, not just one (decided after explicit reasoning, not by
default): each catches a different failure mode the others miss. Dropping
any one weakens a different part of the validation story, and computing
all four is cheap (closed-form on arrays already in memory) — there is no
real cost to reporting the full battery, only a compute-free presentation
choice.

Indices computed: NDVI, NDWI (McFeeters), MNDWI, BSI.
Band order assumed (must match module2_dataset.py's OPTICAL_BAND_ORDER):
    0: B2 (Blue)   1: B3 (Green)   2: B4 (Red)
    3: B8 (NIR)    4: B11 (SWIR1)  5: B12 (SWIR2)

TODO: confirm this band order matches what your GEE export actually
produces — a swapped index here would silently corrupt every reported
number, exactly the failure mode flagged repeatedly in this project before.
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import r2_score

BAND_IDX = {"B2": 0, "B3": 1, "B4": 2, "B8": 3, "B11": 4, "B12": 5}
EPS = 1e-6

# Pixel size in metres — needed to convert pixel-position error into a
# real shoreline error in metres. Sentinel-2 native = 10m.
# TODO: confirm this matches your actual GEE export resolution.
PIXEL_SIZE_M = 10.0


def compute_ndvi(optical):
    nir = optical[..., BAND_IDX["B8"], :, :]
    red = optical[..., BAND_IDX["B4"], :, :]
    return (nir - red) / (nir + red + EPS)


def compute_ndwi(optical):
    """McFeeters NDWI: (Green - NIR) / (Green + NIR)"""
    green = optical[..., BAND_IDX["B3"], :, :]
    nir = optical[..., BAND_IDX["B8"], :, :]
    return (green - nir) / (green + nir + EPS)


def compute_mndwi(optical):
    """Modified NDWI: (Green - SWIR1) / (Green + SWIR1) — better for
    turbid/built-up coastal water than standard NDWI."""
    green = optical[..., BAND_IDX["B3"], :, :]
    swir1 = optical[..., BAND_IDX["B11"], :, :]
    return (green - swir1) / (green + swir1 + EPS)


def compute_bsi(optical):
    """Bare Soil Index: ((SWIR1+Red) - (NIR+Blue)) / ((SWIR1+Red) + (NIR+Blue))"""
    swir1 = optical[..., BAND_IDX["B11"], :, :]
    red = optical[..., BAND_IDX["B4"], :, :]
    nir = optical[..., BAND_IDX["B8"], :, :]
    blue = optical[..., BAND_IDX["B2"], :, :]
    return ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + EPS)


def to_numpy(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t


def rescale_index_to_01(index_arr):
    """Indices range [-1,1] — rescale to [0,1] for SSIM, which expects a
    bounded positive range."""
    return (index_arr + 1.0) / 2.0


def shoreline_position_error(pred_index, gt_index, threshold=0.0, pixel_size_m=PIXEL_SIZE_M):
    """
    Approximates shoreline-position error in metres by comparing the
    water/land boundary (index > threshold = water) row-by-row across the
    patch, measuring pixel offset between predicted and GT boundary, then
    converting to metres. This is a patch-level approximation of the
    transect-based DSAS error used in Paper 1 — for the actual paper,
    re-run this through your DSAS transect pipeline directly on full
    mosaicked scenes, not isolated patches, for the authoritative number.
    This patch-level version is for fast per-batch monitoring during
    training/validation only.
    """
    pred_water = pred_index > threshold
    gt_water = gt_index > threshold

    h, w = gt_water.shape
    row_errors = []
    for r in range(h):
        pred_row = pred_water[r]
        gt_row = gt_water[r]
        pred_edge = np.argmax(pred_row) if pred_row.any() else None
        gt_edge = np.argmax(gt_row) if gt_row.any() else None
        if pred_edge is not None and gt_edge is not None:
            row_errors.append(abs(pred_edge - gt_edge))

    if len(row_errors) == 0:
        return np.nan  # no detectable boundary in this patch — exclude from aggregate, don't treat as zero error
    return float(np.mean(row_errors) * pixel_size_m)


def evaluate_batch(pred_optical, gt_optical):
    """
    pred_optical, gt_optical: (B, 6, H, W) tensors or arrays, normalized [0,1]
    Returns per-sample metric lists (not pre-averaged) so you can report
    mean +/- std, which reviewers expect to see variance, not just a
    single averaged number.
    """
    pred = to_numpy(pred_optical)
    gt = to_numpy(gt_optical)

    results = {
        "image_psnr": [], "image_ssim": [],
        "ndvi_rmse": [], "ndvi_ssim": [], "ndvi_r2": [],
        "ndwi_rmse": [], "ndwi_ssim": [], "ndwi_r2": [],
        "mndwi_rmse": [], "mndwi_ssim": [], "mndwi_r2": [],
        "bsi_rmse": [], "bsi_ssim": [], "bsi_r2": [],
        "ndwi_shoreline_error_m": [],
    }

    batch_size = pred.shape[0]
    for i in range(batch_size):
        p, g = pred[i], gt[i]  # (6, H, W) each

        # --- raw image metrics (comparability with prior S2O / CloudBreaker work) ---
        psnr_vals, ssim_vals = [], []
        for c in range(p.shape[0]):
            psnr_vals.append(psnr(g[c], p[c], data_range=1.0))
            ssim_vals.append(ssim(g[c], p[c], data_range=1.0))
        results["image_psnr"].append(float(np.mean(psnr_vals)))
        results["image_ssim"].append(float(np.mean(ssim_vals)))

        # --- index-level fidelity (the actual novelty contribution) ---
        for name, fn in [("ndvi", compute_ndvi), ("ndwi", compute_ndwi),
                          ("mndwi", compute_mndwi), ("bsi", compute_bsi)]:
            idx_p, idx_g = fn(p), fn(g)

            rmse = float(np.sqrt(np.mean((idx_p - idx_g) ** 2)))
            results[f"{name}_rmse"].append(rmse)

            ssim_idx = float(ssim(rescale_index_to_01(idx_g), rescale_index_to_01(idx_p), data_range=1.0))
            results[f"{name}_ssim"].append(ssim_idx)

            # R^2 over flattened pixels — matches Paper 1's per-transect R^2 convention
            r2 = float(r2_score(idx_g.flatten(), idx_p.flatten()))
            results[f"{name}_r2"].append(r2)

        # --- shoreline position error (NDWI-based, most directly comparable to Paper 1) ---
        shoreline_err = shoreline_position_error(compute_ndwi(p), compute_ndwi(g))
        results["ndwi_shoreline_error_m"].append(shoreline_err)

    return results


def summarize(results_dict):
    """Prints mean +/- std (NaN-aware) for each metric. Use this exact
    table format in the paper for consistency across both papers."""
    summary = {}
    for k, v in results_dict.items():
        arr = np.array(v, dtype=np.float64)
        valid = arr[~np.isnan(arr)]
        n_excluded = len(arr) - len(valid)
        summary[k] = {
            "mean": float(valid.mean()) if len(valid) else float("nan"),
            "std": float(valid.std()) if len(valid) else float("nan"),
            "n": len(valid),
            "n_excluded": n_excluded,
        }
        excl_note = f"  ({n_excluded} excluded, no boundary)" if n_excluded else ""
        print(f"{k:24s}: {summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f}  (n={summary[k]['n']}){excl_note}")
    return summary


if __name__ == "__main__":
    print("=" * 50)
    print("MODULE 6 — Evaluation Suite Verification")
    print("=" * 50)

    torch.manual_seed(42)
    B = 4
    pred = torch.rand(B, 6, 256, 256) * 0.5 + 0.25   # plausible reflectance range
    gt = pred + torch.randn(B, 6, 256, 256) * 0.02   # GT close to pred, small noise
    gt = torch.clamp(gt, 0, 1)

    results = evaluate_batch(pred, gt)
    print("\nMetric summary (synthetic near-identical pred/gt, sanity check only):")
    summarize(results)

    print("\nSanity expectation: RMSE should be small, R^2 should be high "
          "(close to 1), SSIM should be high, since pred and gt are nearly "
          "identical by construction in this synthetic test.")
    print("\n=== Module 6 Complete ===")
