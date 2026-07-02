"""
MODULE 10e — MNDWI Smoothing Diagnostic (last resort before reporting
fragmentation as a stated limitation)
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
PURPOSE: module10d ruled out point-selection logic as the cause of the
81.8m MAE — 73.5% of transect intersections are genuinely MultiPoint,
meaning the GAN-reconstructed water/land boundary is fragmented at
pixel level (individual pixels flipping across MNDWI=0 due to
reconstruction noise), not a script bug. This is the root-cause fix:
smooth the MNDWI raster BEFORE thresholding, to merge spurious
single-pixel boundary noise into one clean shoreline, same as a human
analyst would visually do when reading a noisy classified map.

THIS DOES NOT CHANGE: the model, checkpoint, MNDWI threshold (still
0.0), coastal-strip mask, transect geometry, or MAX_VALID_DIST. It
only adds a smoothing step to the MNDWI array immediately before the
binary water-mask thresholding step — applied IDENTICALLY to both the
GAN output and (for fair comparison) the same real ground-truth
mosaics used in module10b, so this is not "smooth the GAN to make it
look better" — both sides get the same treatment.

TESTS THREE SMOOTHING OPTIONS (report whichever genuinely helps, or
none, honestly):
    A. No smoothing (baseline — replicates module10c's 81.8m result)
    B. Gaussian smoothing (sigma=1 pixel, ~10m)
    C. Binary morphological closing (3x3) on the water mask post-threshold

OUTPUTS:
    training/outputs_module10e_diagnostic/
        smoothing_comparison.csv
        (console: MAE + multipoint-rate for each option, GAN vs GT)
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.merge import merge as rasterio_merge
from rasterio.features import shapes as rio_shapes
from scipy.ndimage import gaussian_filter, binary_closing
from shapely.ops import unary_union
from shapely.geometry import Polygon
import torch

ROOT = r"E:\SAR-Optical-Synthesis"
TRAINING_DIR = os.path.join(ROOT, "training")
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)

from module2_dataset import normalize_optical, read_tif_as_array          # noqa: E402
from module6_evaluation import compute_mndwi                               # noqa: E402
from module10_monsoon_shoreline import (                                   # noqa: E402
    COASTAL_STRIP_COORDS, MAX_VALID_DIST, MNDWI_THRESHOLD,
    SAR_DIR, REF_DIR,
    YEAR_TO_DATE, REF_DATE_FOR_YEAR, AOI_NAME, PATCH_IDS,
    CHECKPOINT, DEVICE,
    regenerate_paper1_transects, load_generator, build_input_tensor,
    compute_signed_distances,
)

GT_DIR = os.path.join(ROOT, "data", "optical_ground_truth")

PAPER1_OUT_DIR = os.path.join(ROOT, "data", "paper1", "outputs")
DSAS_CSV = os.path.join(PAPER1_OUT_DIR, "dsas_transects_classified.csv")
ANNUAL_DIST_CSV = os.path.join(PAPER1_OUT_DIR, "annual_distances_processed.csv")

OUTPUT_DIR = os.path.join(ROOT, "training", "outputs_module10e_diagnostic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = [2021, 2023]
GT_YEARS_DATES = {2021: "2021-06-01", 2023: "2023-06-01"}

SMOOTHING_OPTIONS = ["none", "gaussian", "morph_close"]


def reconstruct_gan_mosaic(generator, year):
    sar_date = YEAR_TO_DATE[year]
    ref_date = REF_DATE_FOR_YEAR[year]
    memfiles, datasets = [], []
    try:
        for pid in PATCH_IDS:
            sar_path = os.path.join(SAR_DIR, f"s1_{AOI_NAME}_{pid}_{sar_date}.tif")
            ref_path = os.path.join(REF_DIR, f"s2ref_{AOI_NAME}_{pid}_{ref_date}.tif")
            if not (os.path.exists(sar_path) and os.path.exists(ref_path)):
                continue
            with rasterio.open(sar_path) as src:
                profile = src.profile.copy()
            input_tensor = build_input_tensor(sar_path, ref_path).to(DEVICE)
            with torch.no_grad():
                pred = generator(input_tensor)
            pred_np = pred.squeeze(0).cpu().numpy().astype(np.float32)
            profile.update(count=6, dtype="float32")
            memfile = rasterio.io.MemoryFile()
            with memfile.open(**profile) as dst:
                dst.write(pred_np)
            memfiles.append(memfile)
            datasets.append(memfile.open())
        mosaic_arr, mosaic_transform = rasterio_merge(datasets)
        mosaic_profile = datasets[0].profile.copy()
        mosaic_profile.update(height=mosaic_arr.shape[1], width=mosaic_arr.shape[2],
                               transform=mosaic_transform)
    finally:
        for ds in datasets:
            ds.close()
        for mf in memfiles:
            mf.close()
    return mosaic_arr, mosaic_profile


def build_gt_mosaic(year):
    gt_date = GT_YEARS_DATES[year]
    memfiles, datasets = [], []
    try:
        for pid in PATCH_IDS:
            gt_path = os.path.join(GT_DIR, f"s2gt_{AOI_NAME}_{pid}_{gt_date}.tif")
            if not os.path.exists(gt_path):
                continue
            arr = read_tif_as_array(gt_path, expected_bands=6)
            arr_norm = normalize_optical(arr).astype(np.float32)
            with rasterio.open(gt_path) as src:
                profile = src.profile.copy()
            profile.update(count=6, dtype="float32")
            memfile = rasterio.io.MemoryFile()
            with memfile.open(**profile) as dst:
                dst.write(arr_norm)
            memfiles.append(memfile)
            datasets.append(memfile.open())
        mosaic_arr, mosaic_transform = rasterio_merge(datasets)
        mosaic_profile = datasets[0].profile.copy()
        mosaic_profile.update(height=mosaic_arr.shape[1], width=mosaic_arr.shape[2],
                               transform=mosaic_transform)
    finally:
        for ds in datasets:
            ds.close()
        for mf in memfiles:
            mf.close()
    return mosaic_arr, mosaic_profile


def extract_shoreline(mosaic_arr, mosaic_profile, smoothing):
    mndwi = compute_mndwi(np.stack([mosaic_arr[i] for i in range(6)], axis=0))
    transform = mosaic_profile["transform"]

    if smoothing == "none":
        water_mask = (mndwi > MNDWI_THRESHOLD).astype(np.uint8)
    elif smoothing == "gaussian":
        mndwi_smooth = gaussian_filter(mndwi, sigma=1.0)
        water_mask = (mndwi_smooth > MNDWI_THRESHOLD).astype(np.uint8)
    elif smoothing == "morph_close":
        water_mask_raw = (mndwi > MNDWI_THRESHOLD).astype(np.uint8)
        water_mask = binary_closing(water_mask_raw, structure=np.ones((3, 3))).astype(np.uint8)
    else:
        raise ValueError(smoothing)

    polygons = []
    for geom, val in rio_shapes(water_mask, transform=transform):
        if val == 1:
            polygons.append(Polygon(geom["coordinates"][0]))
    if not polygons:
        return None

    water_union = unary_union(polygons)
    boundary = water_union.boundary

    shoreline_gdf = gpd.GeoDataFrame(geometry=[boundary], crs="EPSG:32645")
    coastal_strip = Polygon(COASTAL_STRIP_COORDS)
    coastal_strip_gdf = gpd.GeoDataFrame(geometry=[coastal_strip], crs="EPSG:4326").to_crs(epsg=32645)
    return gpd.clip(shoreline_gdf, coastal_strip_gdf)


def signed_distance_with_multipoint_flag(t_line, intersection):
    if intersection.is_empty:
        return np.nan, False
    is_mp = intersection.geom_type == "MultiPoint"
    if is_mp:
        int_pt = list(intersection.geoms)[0]
    elif intersection.geom_type == "Point":
        int_pt = intersection
    else:
        int_pt = intersection.centroid

    t_len = t_line.length
    proj_d = t_line.project(int_pt)
    signed = proj_d - (t_len / 2)
    return (signed if abs(signed) <= MAX_VALID_DIST else np.nan), is_mp


def main():
    print("=" * 60)
    print("  MODULE 10e — MNDWI Smoothing Diagnostic")
    print("=" * 60)

    generator = load_generator(CHECKPOINT)
    transects_gdf = regenerate_paper1_transects()
    dsas_df = pd.read_csv(DSAS_CSV)
    valid_ids = set(dsas_df["TransectID"].astype(int))
    transects_gdf = transects_gdf[transects_gdf["TransectID"].isin(valid_ids)].reset_index(drop=True)
    print(f"Aligned to {len(transects_gdf)} surviving Paper 1 transects\n")

    results = []

    for year in YEARS:
        print(f"--- Year {year}: reconstructing GAN + building GT mosaics ---")
        gan_mosaic_arr, gan_profile = reconstruct_gan_mosaic(generator, year)
        gt_mosaic_arr, gt_profile = build_gt_mosaic(year)

        for smoothing in SMOOTHING_OPTIONS:
            gan_shoreline_gdf = extract_shoreline(gan_mosaic_arr, gan_profile, smoothing)
            gt_shoreline_gdf = extract_shoreline(gt_mosaic_arr, gt_profile, smoothing)

            if gan_shoreline_gdf is None or gt_shoreline_gdf is None or \
               gan_shoreline_gdf.empty or gt_shoreline_gdf.empty:
                print(f"  [{year}/{smoothing}] empty shoreline, skipping")
                continue

            gan_line = unary_union(gan_shoreline_gdf.geometry.values)
            gt_line = unary_union(gt_shoreline_gdf.geometry.values)

            for _, t_row in transects_gdf.iterrows():
                tid = t_row["TransectID"]
                t_line = t_row.geometry

                gan_int = t_line.intersection(gan_line)
                gt_int = t_line.intersection(gt_line)

                gan_dist, gan_mp = signed_distance_with_multipoint_flag(t_line, gan_int)
                gt_dist, gt_mp = signed_distance_with_multipoint_flag(t_line, gt_int)

                results.append({
                    "Year": year,
                    "Smoothing": smoothing,
                    "TransectID": tid,
                    "GAN_dist_m": gan_dist,
                    "GT_dist_m": gt_dist,
                    "GAN_is_multipoint": gan_mp,
                    "abs_error_m": (abs(gan_dist - gt_dist)
                                    if not (np.isnan(gan_dist) or np.isnan(gt_dist))
                                    else np.nan),
                })

    df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "smoothing_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}\n")

    print("=" * 60)
    print("  SMOOTHING COMPARISON SUMMARY (GAN vs Ground Truth)")
    print("=" * 60)
    for smoothing in SMOOTHING_OPTIONS:
        sub = df[df["Smoothing"] == smoothing].dropna(subset=["abs_error_m"])
        mp_rate = df[df["Smoothing"] == smoothing]["GAN_is_multipoint"].mean() * 100
        if len(sub) == 0:
            print(f"\n{smoothing:14s}: no valid observations")
            continue
        print(f"\n{smoothing:14s}: MAE = {sub['abs_error_m'].mean():.2f} +/- "
              f"{sub['abs_error_m'].std():.2f} m  (n={len(sub)})  "
              f"| GAN multipoint rate = {mp_rate:.1f}%")

    print("\n" + "=" * 60)
    print("  DECISION GUIDE")
    print("=" * 60)
    print("""
  Compare each smoothing option's MAE to the 'none' baseline
  (should reproduce ~81.8m from module10c, modulo small differences
  from this script independently re-running inference).

  If gaussian or morph_close meaningfully LOWERS the MAE and the
  multipoint rate -> legitimate fix, adopt it (re-run module10/10c
  with this smoothing step added to extract_mndwi_shoreline(), report
  the improved number).

  If neither helps -> fragmentation is intrinsic to the GAN's
  pixel-level output, not fixable by post-hoc smoothing without
  losing real signal. Report 81.8m MAE / 73.5% multipoint rate as a
  stated, well-understood limitation and move on with confidence —
  you will have tested the two most reasonable fixes (point-selection,
  smoothing) and both will have failed to help, which is itself a
  complete and honest diagnostic story for the paper.
""")


if __name__ == "__main__":
    main()