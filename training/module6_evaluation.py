"""
MODULE 6 — Evaluation Suite
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
Reports ALL FOUR metric families on the held-out test set:
  - R^2            (computed globally across all test pixels, not per-patch
                     — per-patch R^2 is numerically unstable for spatially
                     homogeneous patches where variance is near zero)
  - RMSE           (absolute index-unit error)
  - SSIM-on-index  (spatial-pattern fidelity of the index map)
  - Shoreline error (metres)

Band order (must match module2_dataset.py OPTICAL_BAND_ORDER):
    0: B2 (Blue)   1: B3 (Green)   2: B4 (Red)
    3: B8 (NIR)    4: B11 (SWIR1)  5: B12 (SWIR2)
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import r2_score

BAND_IDX = {"B2": 0, "B3": 1, "B4": 2, "B8": 3, "B11": 4, "B12": 5}
EPS = 1e-6
PIXEL_SIZE_M = 10.0


def compute_ndvi(optical):
    nir = optical[..., BAND_IDX["B8"], :, :]
    red = optical[..., BAND_IDX["B4"], :, :]
    return (nir - red) / (nir + red + EPS)


def compute_ndwi(optical):
    green = optical[..., BAND_IDX["B3"], :, :]
    nir = optical[..., BAND_IDX["B8"], :, :]
    return (green - nir) / (green + nir + EPS)


def compute_mndwi(optical):
    green = optical[..., BAND_IDX["B3"], :, :]
    swir1 = optical[..., BAND_IDX["B11"], :, :]
    return (green - swir1) / (green + swir1 + EPS)


def compute_bsi(optical):
    swir1 = optical[..., BAND_IDX["B11"], :, :]
    red = optical[..., BAND_IDX["B4"], :, :]
    nir = optical[..., BAND_IDX["B8"], :, :]
    blue = optical[..., BAND_IDX["B2"], :, :]
    return ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + EPS)


def to_numpy(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t


def rescale_index_to_01(index_arr):
    return (index_arr + 1.0) / 2.0


def shoreline_position_error(pred_index, gt_index, threshold=0.0, pixel_size_m=PIXEL_SIZE_M):
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
        return np.nan
    return float(np.mean(row_errors) * pixel_size_m)


def evaluate_batch(pred_optical, gt_optical):
    """
    pred_optical, gt_optical: (B, 6, H, W) tensors or arrays, normalized [0,1]
    Returns per-sample metric lists for RMSE, SSIM, shoreline error,
    and accumulated pixel arrays for global R^2 computation in summarize().
    R^2 is NOT computed per-patch — it is computed globally across all
    test pixels in summarize() to avoid numerical instability from
    low-variance homogeneous patches.
    """
    pred = to_numpy(pred_optical)
    gt = to_numpy(gt_optical)

    results = {
        "image_psnr": [], "image_ssim": [],
        "ndvi_rmse": [], "ndvi_ssim": [],
        "ndvi_r2_pred": [], "ndvi_r2_gt": [],
        "ndwi_rmse": [], "ndwi_ssim": [],
        "ndwi_r2_pred": [], "ndwi_r2_gt": [],
        "mndwi_rmse": [], "mndwi_ssim": [],
        "mndwi_r2_pred": [], "mndwi_r2_gt": [],
        "bsi_rmse": [], "bsi_ssim": [],
        "bsi_r2_pred": [], "bsi_r2_gt": [],
        "ndwi_shoreline_error_m": [],
    }

    batch_size = pred.shape[0]
    for i in range(batch_size):
        p, g = pred[i], gt[i]  # (6, H, W)

        # raw image metrics
        psnr_vals, ssim_vals = [], []
        for c in range(p.shape[0]):
            psnr_vals.append(psnr(g[c], p[c], data_range=1.0))
            ssim_vals.append(ssim(g[c], p[c], data_range=1.0))
        results["image_psnr"].append(float(np.mean(psnr_vals)))
        results["image_ssim"].append(float(np.mean(ssim_vals)))

        # index metrics
        for name, fn in [("ndvi", compute_ndvi), ("ndwi", compute_ndwi),
                          ("mndwi", compute_mndwi), ("bsi", compute_bsi)]:
            idx_p, idx_g = fn(p), fn(g)

            rmse = float(np.sqrt(np.mean((idx_p - idx_g) ** 2)))
            results[f"{name}_rmse"].append(rmse)

            ssim_idx = float(ssim(rescale_index_to_01(idx_g),
                                   rescale_index_to_01(idx_p), data_range=1.0))
            results[f"{name}_ssim"].append(ssim_idx)

            # accumulate pixels for global R^2 — NOT computed per patch
            results[f"{name}_r2_pred"].extend(idx_p.flatten().tolist())
            results[f"{name}_r2_gt"].extend(idx_g.flatten().tolist())

        # shoreline error
        shoreline_err = shoreline_position_error(compute_ndwi(p), compute_ndwi(g))
        results["ndwi_shoreline_error_m"].append(shoreline_err)

    return results


def summarize(results_dict):
    """
    Computes final metrics. R^2 is computed globally from accumulated
    pixel arrays. All other metrics are mean +/- std across samples.
    """
    summary = {}
    index_names = ["ndvi", "ndwi", "mndwi", "bsi"]

    scalar_keys = (
        ["image_psnr", "image_ssim"] +
        [f"{n}_{m}" for n in index_names for m in ["rmse", "ssim"]] +
        ["ndwi_shoreline_error_m"]
    )

    for k in scalar_keys:
        arr = np.array(results_dict[k], dtype=np.float64)
        valid = arr[~np.isnan(arr)]
        n_excluded = len(arr) - len(valid)
        summary[k] = {
            "mean": float(valid.mean()) if len(valid) else float("nan"),
            "std": float(valid.std()) if len(valid) else float("nan"),
            "n": len(valid),
        }
        excl_note = f"  ({n_excluded} excluded, no boundary)" if n_excluded else ""
        print(f"{k:28s}: {summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f}"
              f"  (n={summary[k]['n']}){excl_note}")

    # global R^2 across all test pixels
    for name in index_names:
        gt_all = np.array(results_dict[f"{name}_r2_gt"], dtype=np.float64)
        pred_all = np.array(results_dict[f"{name}_r2_pred"], dtype=np.float64)
        r2 = float(r2_score(gt_all, pred_all))
        summary[f"{name}_r2"] = {"mean": r2, "std": float("nan"), "n": len(gt_all)}
        print(f"{name}_r2 (global)           : {r2:.4f}  (n_pixels={len(gt_all)})")

    return summary


if __name__ == "__main__":
    print("=" * 50)
    print("MODULE 6 — Evaluation Suite Verification")
    print("=" * 50)

    torch.manual_seed(42)
    B = 4
    pred = torch.rand(B, 6, 256, 256) * 0.5 + 0.25
    gt = pred + torch.randn(B, 6, 256, 256) * 0.02
    gt = torch.clamp(gt, 0, 1)

    results = evaluate_batch(pred, gt)
    print("\nMetric summary (synthetic near-identical pred/gt):")
    summarize(results)
    print("\nExpected: RMSE small, R^2 close to 1, SSIM high.")
    print("\n=== Module 6 Complete ===")