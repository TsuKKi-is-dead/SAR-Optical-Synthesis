"""
MODULE 10b — Ground-Truth Diagnostic
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
PURPOSE: module10 found a coastline-wide landward shift at monsoon
dates (77% of transects landward vs. 39.6% post-monsoon; Stable and
High Accretion transects showing displacement comparable to or larger
than High Erosion transects) and a high MAX_VALID_DIST exclusion rate
(35.6%). Before reporting this as a real monsoon shoreline finding,
this script isolates whether it is a genuine hydrological signal or a
GAN reconstruction artifact, by running the IDENTICAL MNDWI ->
shoreline -> transect-displacement pipeline on REAL ground-truth
optical patches (not GAN output) for the two years that have them
(2021, 2023 — confirmed has_gt=True in manifest.csv; 2022 has no GT).

LOGIC:
    If real ground truth ALSO shows a large coastline-wide landward
    shift + high exclusion rate -> genuine monsoon hydrological signal,
    application section is legitimate (and strengthened — validated
    against real data, not just GAN output).

    If real ground truth looks calm (small, erosion-class-correlated
    displacement, low exclusion rate) while GAN output showed the large
    uniform shift -> reconstruction artifact, do NOT report the GAN
    displacement numbers as a real finding.

Reuses module10's transect regeneration, signed-distance, and locked
constants verbatim — only the optical source (ground truth vs. GAN
output) differs.
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.merge import merge as rasterio_merge
from rasterio.features import shapes as rio_shapes
from shapely.ops import unary_union
from shapely.geometry import Polygon

ROOT = r"E:\SAR-Optical-Synthesis"
GT_DIR = os.path.join(ROOT, "data", "optical_ground_truth")
MANIFEST_CSV = os.path.join(ROOT, "data", "manifest.csv")

PAPER1_DIR = os.path.join(ROOT, "data", "paper1")
PAPER1_OUT_DIR = os.path.join(PAPER1_DIR, "outputs")
SHORELINE_2013 = os.path.join(PAPER1_DIR, "shoreline_2013.shp")
DSAS_CSV = os.path.join(PAPER1_OUT_DIR, "dsas_transects_classified.csv")
ANNUAL_DIST_CSV = os.path.join(PAPER1_OUT_DIR, "annual_distances_processed.csv")

OUTPUT_DIR = os.path.join(ROOT, "training", "outputs_module10b_diagnostic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAINING_DIR = os.path.join(ROOT, "training")
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)

from module2_dataset import normalize_optical, read_tif_as_array   # noqa: E402
from module6_evaluation import compute_mndwi                        # noqa: E402

# Import the regeneration + distance logic from module10 directly —
# do not reimplement, avoid any drift between the two scripts.
from module10_monsoon_shoreline import (                            # noqa: E402
    COASTAL_STRIP_COORDS, MAX_VALID_DIST, MNDWI_THRESHOLD,
    regenerate_paper1_transects, compute_signed_distances,
    OPTICAL_BAND_ORDER, B2_IDX, B3_IDX, B4_IDX, B11_IDX,
)

AOI_NAME = "brahmapur"
PATCH_IDS = [f"{i:04d}" for i in range(72)]

# Only years with real ground truth (confirmed earlier: 2021 and 2023
# have has_gt=True rows; 2022 does not).
GT_YEARS = {
    2021: "2021-06-01",
    2023: "2023-06-01",
}


def load_gt_mosaic(year):
    gt_date = GT_YEARS[year]
    datasets = []
    memfiles = []
    missing = []
    try:
        for pid in PATCH_IDS:
            gt_path = os.path.join(GT_DIR, f"s2gt_{AOI_NAME}_{pid}_{gt_date}.tif")
            if not os.path.exists(gt_path):
                missing.append(pid)
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

        if missing:
            print(f"  WARNING [{year}]: {len(missing)} GT patches missing: {missing}")

        if not datasets:
            raise RuntimeError(f"Year {year}: no ground-truth patches found at all.")

        mosaic_arr, mosaic_transform = rasterio_merge(datasets)
        mosaic_profile = datasets[0].profile.copy()
        mosaic_profile.update(
            height=mosaic_arr.shape[1],
            width=mosaic_arr.shape[2],
            transform=mosaic_transform,
        )
    finally:
        for ds in datasets:
            ds.close()
        for mf in memfiles:
            mf.close()

    print(f"Year {year}: GT mosaic built from {len(datasets)}/{len(PATCH_IDS)} patches, "
          f"shape={mosaic_arr.shape}")
    return mosaic_arr, mosaic_profile


def extract_shoreline_from_mosaic(mosaic_arr, mosaic_profile):
    mndwi = compute_mndwi(np.stack([mosaic_arr[i] for i in range(6)], axis=0))
    water_mask = (mndwi > MNDWI_THRESHOLD).astype(np.uint8)

    transform = mosaic_profile["transform"]
    polygons = []
    for geom, val in rio_shapes(water_mask, transform=transform):
        if val == 1:
            polygons.append(Polygon(geom["coordinates"][0]))

    if not polygons:
        raise RuntimeError("No water polygons found in GT mosaic — check MNDWI threshold.")

    water_union = unary_union(polygons)
    boundary = water_union.boundary

    shoreline_gdf = gpd.GeoDataFrame(geometry=[boundary], crs="EPSG:32645")
    coastal_strip = Polygon(COASTAL_STRIP_COORDS)
    coastal_strip_gdf = gpd.GeoDataFrame(geometry=[coastal_strip], crs="EPSG:4326").to_crs(epsg=32645)
    shoreline_clipped = gpd.clip(shoreline_gdf, coastal_strip_gdf)

    return mndwi, shoreline_clipped


def main():
    print("=" * 60)
    print("  MODULE 10b — Ground-Truth Diagnostic")
    print("  (real optical, NOT GAN output — isolating artifact vs. signal)")
    print("=" * 60)

    dsas_df = pd.read_csv(DSAS_CSV)
    annual_df = pd.read_csv(ANNUAL_DIST_CSV)
    transects_gdf = regenerate_paper1_transects()

    valid_ids = set(dsas_df["TransectID"].astype(int))
    transects_gdf = transects_gdf[transects_gdf["TransectID"].isin(valid_ids)].reset_index(drop=True)
    print(f"Aligned to {len(transects_gdf)} surviving Paper 1 transects\n")

    all_rows = []
    for year in GT_YEARS:
        mosaic_arr, mosaic_profile = load_gt_mosaic(year)
        mndwi, shoreline_gdf = extract_shoreline_from_mosaic(mosaic_arr, mosaic_profile)

        if shoreline_gdf.empty:
            print(f"  Year {year}: shoreline extraction empty after clip, skipping.")
            continue

        shoreline_line = unary_union(shoreline_gdf.geometry.values)
        gt_dist = compute_signed_distances(transects_gdf, shoreline_line)

        postmonsoon_col = f"dist_{year}"
        for i, tid in enumerate(transects_gdf["TransectID"].values):
            pm_row = annual_df.loc[annual_df["TransectID"] == tid, postmonsoon_col]
            pm_dist = pm_row.values[0] if len(pm_row) else np.nan
            cls_row = dsas_df.loc[dsas_df["TransectID"] == tid, "Class"]
            cls = cls_row.values[0] if len(cls_row) else "Unknown"

            all_rows.append({
                "TransectID": tid,
                "Year": year,
                "GT_dist_m": gt_dist[i],
                "PostMonsoon_dist_m": pm_dist,
                "Seasonal_displacement_m": (
                    gt_dist[i] - pm_dist
                    if not (np.isnan(gt_dist[i]) or np.isnan(pm_dist))
                    else np.nan
                ),
                "Class": cls,
            })

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(OUTPUT_DIR, "groundtruth_seasonal_displacement.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    valid = df.dropna(subset=["Seasonal_displacement_m"])
    n_excluded = len(df) - len(valid)
    print("\n" + "=" * 60)
    print("  GROUND-TRUTH DIAGNOSTIC RESULTS")
    print("=" * 60)
    print(f"Valid observations: {len(valid)} / {len(df)} "
          f"({n_excluded} excluded by MAX_VALID_DIST={MAX_VALID_DIST}m filter)")
    print(f"  (compare to GAN run: 127/357 excluded = 35.6%)")

    by_year = valid.groupby("Year")["Seasonal_displacement_m"].agg(["mean", "std", "count"])
    print("\nMean seasonal displacement per year, REAL ground truth (m):")
    print(by_year.to_string())

    by_class = valid.groupby("Class")["Seasonal_displacement_m"].agg(
        mean_displacement="mean", std_displacement="std", n="count"
    )
    print("\nDisplacement by Paper 1 erosion class (REAL ground truth):")
    print(by_class.to_string())

    pct_landward_gt = (valid["GT_dist_m"] < 0).mean() * 100
    pct_landward_pm = (valid["PostMonsoon_dist_m"] < 0).mean() * 100
    print(f"\n% transects landward-of-baseline (GT monsoon-onset): {pct_landward_gt:.1f}%")
    print(f"% transects landward-of-baseline (post-monsoon):      {pct_landward_pm:.1f}%")
    print("  (compare to GAN run: 77.0% vs 39.6%)")

    print("\n" + "=" * 60)
    print("  INTERPRETATION GUIDE (not automated — read the numbers above)")
    print("=" * 60)
    print("""
  If the REAL ground-truth numbers above show a similarly large,
  coastline-wide landward shift and a similarly high exclusion rate
  to the GAN run -> this is a genuine monsoon-onset hydrological
  signal (e.g. higher water levels, turbid runoff expanding the MNDWI
  water class even where the true coastline hasn't moved much) and the
  GAN result is plausible, not an artifact.

  If the REAL ground-truth numbers are much smaller / much closer to
  post-monsoon / show low exclusion -> the GAN run's large uniform
  shift is a reconstruction artifact (likely the GAN over-predicting
  water-like spectral signatures under monsoon SAR conditions), and
  the seasonal_displacement_table.csv from module10 should NOT be
  reported as a real shoreline-position finding without further
  investigation (e.g. checking whether the GAN systematically biases
  B3/B11 under monsoon SAR inputs specifically).
""")


if __name__ == "__main__":
    main()