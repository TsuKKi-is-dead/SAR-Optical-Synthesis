"""
MODULE 11 — Monsoon LULC Classification (Paper 2 Application Section)
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
Applies Paper 1's VERIFIED MNDWI two-threshold classification scheme
(Land / Intertidal / Water) to the GAN-reconstructed monsoon optical
mosaics, producing the first monsoon-season LULC maps for this
coastline — directly comparable to Paper 1's post-monsoon 2021/2024
LULC maps using the IDENTICAL classification method.

THRESHOLD PROVENANCE (do not change — verified, not assumed):
    Reverse-engineered from data/paper1/accuracy_verification_Final1.csv
    (180 points, Point_ID/Longitude/Latitude/MNDWI/Predicted_Class/
    True_Class). Predicted_Class shows ZERO overlap in MNDWI value
    between classes:
        Land max MNDWI       = -0.1709
        Intertidal min MNDWI =  0.0086
        Intertidal max MNDWI =  0.1953
        Water min MNDWI      =  0.2197
    Thresholds below are the midpoints of these two gaps. They
    reproduce Predicted_Class for all 180/180 verification points, and
    Predicted_Class vs True_Class agreement at these thresholds is
    169/180 = 93.89% — matching Paper 1's reported OA=93.9% almost
    exactly, confirming this verification file IS the source of Paper
    1's reported accuracy and this IS the method Paper 1 used (not a
    separately-trained RF classifier, despite this module's working
    title — see conversation history; the original plan to train an RF
    in Python was deliberately replaced with this threshold approach to
    keep Paper 1 and Paper 2's LULC methodology identical, per explicit
    decision).

METHODOLOGY CONSISTENCY: same locked coastal-strip mask, same MNDWI
formula (module6, reused), same pixel area assumption (10m x 10m,
matching Paper 1's PIXEL_AREA_M2=100) as Paper 1's Module 8.

OUTPUTS (matching Paper 1's Module 8 style):
    training/outputs_module11_lulc/
        lulc_monsoon_<year>.tif            (per-year classified raster)
        fig_lulc_monsoon_maps.png          (side-by-side monsoon LULC, 3 years)
        fig_lulc_monsoon_vs_postmonsoon.png (monsoon 2021 vs Paper 1's 2021 post-monsoon)
        table_lulc_monsoon_area_stats.csv  (class areas in km^2, all 3 monsoon years)
        table_lulc_accuracy_check.csv      (threshold reproduction sanity check)
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.merge import merge as rasterio_merge
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from shapely.geometry import Polygon
import torch

ROOT = r"E:\SAR-Optical-Synthesis"
TRAINING_DIR = os.path.join(ROOT, "training")
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)

from module6_evaluation import compute_mndwi                                # noqa: E402
from module10_monsoon_shoreline import (                                    # noqa: E402
    SAR_DIR, REF_DIR, YEAR_TO_DATE, REF_DATE_FOR_YEAR, AOI_NAME, PATCH_IDS,
    CHECKPOINT, DEVICE, load_generator, build_input_tensor,
    COASTAL_STRIP_COORDS,
)

PAPER1_DIR = os.path.join(ROOT, "data", "paper1")
LULC_2021_PATH = os.path.join(PAPER1_DIR, "LULC_2021_Ganjam.tif")

OUTPUT_DIR = os.path.join(ROOT, "training", "outputs_module11_lulc")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── VERIFIED thresholds (see module docstring for derivation) ──────────────
MNDWI_LAND_INTERTIDAL_THRESHOLD = -0.0811
MNDWI_INTERTIDAL_WATER_THRESHOLD = 0.2075

CLASS_NAMES = {0: "Land", 1: "Intertidal", 2: "Water"}
CLASS_COLORS = {0: "#8B4513", 1: "#F4A460", 2: "#1E90FF"}
PIXEL_AREA_M2 = 100  # 10m x 10m, matches Paper 1's Module 8

YEARS = [2021, 2022, 2023]


def classify_mndwi(mndwi_arr):
    """Reproduces Paper 1's verified threshold rule exactly.
    Returns int16 array: 0=Land, 1=Intertidal, 2=Water."""
    classified = np.full_like(mndwi_arr, 1, dtype=np.int16)  # default Intertidal
    classified[mndwi_arr < MNDWI_LAND_INTERTIDAL_THRESHOLD] = 0
    classified[mndwi_arr > MNDWI_INTERTIDAL_WATER_THRESHOLD] = 2
    return classified


def build_coastal_strip_mask(profile):
    """Builds a boolean mask, True = OUTSIDE the coastal strip, aligned
    to the given raster's transform/shape. Used to restrict area-stats
    computation to the same coastal corridor Paper 1's Module 8 used —
    WITHOUT this, area stats are computed over the full ~470 km^2
    mosaic bounding box instead of the actual coastal AOI, which is not
    comparable to Paper 1's reported areas (single-to-tens of km^2)."""
    coastal_strip = Polygon(COASTAL_STRIP_COORDS)
    coastal_strip_gdf = gpd.GeoDataFrame(geometry=[coastal_strip], crs="EPSG:4326").to_crs(epsg=32645)

    out_shape = (profile["height"], profile["width"])
    transform = profile["transform"]

    # geometry_mask returns True for pixels OUTSIDE the geometry by default
    # (invert=False), which is exactly the "outside AOI" mask we want.
    outside_mask = geometry_mask(
        coastal_strip_gdf.geometry.values,
        out_shape=out_shape,
        transform=transform,
        invert=False,
    )
    return outside_mask


def reconstruct_and_mosaic(generator, year):
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


def verify_thresholds_against_csv():
    """Re-checks the threshold rule against the verification CSV at
    runtime, so any future edit to the threshold constants above is
    caught immediately rather than silently drifting from the verified
    values."""
    csv_path = os.path.join(PAPER1_DIR, "accuracy_verification_Final1.csv")
    df = pd.read_csv(csv_path, skiprows=1)  # skip stray "accuracy_verification" header line

    def classify_row(mndwi):
        if mndwi < MNDWI_LAND_INTERTIDAL_THRESHOLD:
            return "Land"
        elif mndwi > MNDWI_INTERTIDAL_WATER_THRESHOLD:
            return "Water"
        else:
            return "Intertidal"

    df["Reproduced_Class"] = df["MNDWI"].apply(classify_row)
    match_rate = (df["Reproduced_Class"] == df["Predicted_Class"]).mean()

    out_path = os.path.join(OUTPUT_DIR, "table_lulc_accuracy_check.csv")
    df.to_csv(out_path, index=False)

    print(f"Threshold reproduction check: {match_rate*100:.2f}% match against "
          f"Predicted_Class (expect 100.00%)")
    if match_rate < 1.0:
        print("  WARNING: thresholds no longer perfectly reproduce Paper 1's "
              "verification file — investigate before trusting downstream results.")
    return match_rate


def main():
    print("=" * 60)
    print("  MODULE 11 — Monsoon LULC Classification")
    print("  Method: Paper 1's verified MNDWI threshold scheme (NOT RF)")
    print("=" * 60)

    match_rate = verify_thresholds_against_csv()
    print()

    generator = load_generator(CHECKPOINT)

    mosaics = {}
    classified = {}
    area_rows = []

    for year in YEARS:
        print(f"--- Year {year}: reconstructing + classifying ---")
        mosaic_arr, mosaic_profile = reconstruct_and_mosaic(generator, year)
        mndwi = compute_mndwi(np.stack([mosaic_arr[i] for i in range(6)], axis=0))
        classified_arr = classify_mndwi(mndwi)

        # Build coastal-strip mask aligned to THIS mosaic's grid (each
        # year's mosaic can have a slightly different shape/transform
        # depending on which patches were available).
        outside_aoi_mask = build_coastal_strip_mask(mosaic_profile)
        n_outside = outside_aoi_mask.sum()
        n_total = outside_aoi_mask.size
        print(f"  Coastal-strip mask: {n_total - n_outside:,}/{n_total:,} pixels inside AOI "
              f"({(n_total - n_outside) / n_total * 100:.1f}%)")

        # Masked copy for area-stats only — full-extent classified_arr
        # is still saved/plotted as-is for visualization context.
        classified_masked = np.where(outside_aoi_mask, -1, classified_arr)

        mosaics[year] = (mosaic_arr, mosaic_profile)
        classified[year] = (classified_arr, mosaic_profile)

        lulc_path = os.path.join(OUTPUT_DIR, f"lulc_monsoon_{year}.tif")
        lulc_profile = mosaic_profile.copy()
        lulc_profile.update(count=1, dtype="int16", nodata=-1)
        with rasterio.open(lulc_path, "w", **lulc_profile) as dst:
            dst.write(classified_arr[np.newaxis, :, :])
        print(f"  Saved: {lulc_path}  (full mosaic extent, NOT coastal-strip-clipped)")

        lulc_clipped_path = os.path.join(OUTPUT_DIR, f"lulc_monsoon_{year}_coastal_strip.tif")
        with rasterio.open(lulc_clipped_path, "w", **lulc_profile) as dst:
            dst.write(classified_masked[np.newaxis, :, :])
        print(f"  Saved: {lulc_clipped_path}  (coastal-strip-clipped, USE THIS FOR AREA STATS)")

        for cls_id, cls_name in CLASS_NAMES.items():
            px = int((classified_masked == cls_id).sum())  # masked, not full extent
            km2 = px * PIXEL_AREA_M2 / 1e6
            area_rows.append({"Year": year, "Class": cls_name, "Pixels": px, "Area_km2": round(km2, 3)})

    area_df = pd.DataFrame(area_rows)
    area_path = os.path.join(OUTPUT_DIR, "table_lulc_monsoon_area_stats.csv")
    area_df.to_csv(area_path, index=False)
    print(f"\nSaved: {area_path}")
    print("\nMonsoon LULC area statistics (km^2) — COASTAL STRIP ONLY, comparable to Paper 1's Module 8:")
    print(area_df.pivot(index="Year", columns="Class", values="Area_km2").to_string())

    # ── Figure: side-by-side monsoon LULC maps, 3 years ─────────────────────
    cmap_lulc = ListedColormap([CLASS_COLORS[0], CLASS_COLORS[1], CLASS_COLORS[2]])
    norm_lulc = BoundaryNorm([0, 1, 2, 3], 3)

    fig, axes = plt.subplots(1, len(YEARS), figsize=(6 * len(YEARS), 7), facecolor="#F0F4F8")
    fig.suptitle("Monsoon-Season LULC Classification (GAN-Reconstructed)\n"
                 "Brahmapur Coastline, Ganjam, Odisha", fontsize=13, fontweight="bold", y=0.98)

    for ax, year in zip(axes, YEARS):
        classified_arr, mosaic_profile = classified[year]
        bounds = rasterio.transform.array_bounds(
            classified_arr.shape[0], classified_arr.shape[1], mosaic_profile["transform"]
        )
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
        ax.imshow(classified_arr, cmap=cmap_lulc, norm=norm_lulc, extent=extent,
                  aspect="auto", interpolation="nearest")
        ax.set_title(f"{YEAR_TO_DATE[year]} (monsoon)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Easting (m, UTM 45N)", fontsize=9)
        ax.set_ylabel("Northing (m, UTM 45N)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

        # Annotations use the SAME coastal-strip-masked stats as the
        # table (area_df), not the full mosaic extent shown in the
        # background image — keeps figure and table numbers consistent.
        year_stats = area_df[area_df["Year"] == year]
        total_aoi_px = year_stats["Pixels"].sum()
        for cls_id, cls_name in CLASS_NAMES.items():
            row = year_stats[year_stats["Class"] == cls_name]
            km2 = row["Area_km2"].values[0] if len(row) else 0.0
            px = row["Pixels"].values[0] if len(row) else 0
            pct = (px / total_aoi_px * 100) if total_aoi_px > 0 else 0.0
            ax.annotate(f"{cls_name}: {km2:.2f} km^2 ({pct:.1f}%)",
                        xy=(0.02, 0.97 - cls_id * 0.06), xycoords="axes fraction",
                        fontsize=8, color="white",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=CLASS_COLORS[cls_id], alpha=0.85))
        ax.annotate("(stats: coastal strip only; map: full mosaic extent)",
                    xy=(0.02, 0.97 - 3 * 0.06), xycoords="axes fraction",
                    fontsize=6.5, color="black", style="italic")

    legend_patches = [mpatches.Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(3)]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig_path = os.path.join(OUTPUT_DIR, "fig_lulc_monsoon_maps.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {fig_path}")

    # ── Figure: 2021 monsoon vs Paper 1's 2021 post-monsoon, side by side ──
    if os.path.exists(LULC_2021_PATH):
        with rasterio.open(LULC_2021_PATH) as src:
            postmonsoon_2021 = src.read(1).astype(np.int16)
            pm_bounds = src.bounds
            pm_nodata = src.nodata

        if pm_nodata is not None:
            postmonsoon_2021 = np.where(postmonsoon_2021 == pm_nodata, -1, postmonsoon_2021)

        fig, axes = plt.subplots(1, 2, figsize=(13, 7), facecolor="#F0F4F8")
        fig.suptitle("Seasonal LULC Comparison — Brahmapur Coastline, 2021",
                     fontsize=13, fontweight="bold", y=0.98)

        masked_pm = np.ma.masked_where(postmonsoon_2021 < 0, postmonsoon_2021)
        axes[0].imshow(masked_pm, cmap=cmap_lulc, norm=norm_lulc,
                       extent=[pm_bounds.left, pm_bounds.right, pm_bounds.bottom, pm_bounds.top],
                       aspect="auto", interpolation="nearest")
        axes[0].set_title("Post-monsoon 2021 (Paper 1, real Sentinel-2)", fontsize=11)

        classified_2021, profile_2021 = classified[2021]
        bounds_2021 = rasterio.transform.array_bounds(
            classified_2021.shape[0], classified_2021.shape[1], profile_2021["transform"]
        )
        axes[1].imshow(classified_2021, cmap=cmap_lulc, norm=norm_lulc,
                       extent=[bounds_2021[0], bounds_2021[2], bounds_2021[1], bounds_2021[3]],
                       aspect="auto", interpolation="nearest")
        axes[1].set_title(f"Monsoon-onset {YEAR_TO_DATE[2021]} (Paper 2, GAN-reconstructed)", fontsize=11)

        for ax in axes:
            ax.set_xlabel("Longitude / Easting", fontsize=9)
            ax.set_ylabel("Latitude / Northing", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

        legend_patches = [mpatches.Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(3)]
        fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=10,
                   framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig_path2 = os.path.join(OUTPUT_DIR, "fig_lulc_monsoon_vs_postmonsoon.png")
        plt.savefig(fig_path2, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path2}")

        print("\nNOTE: 2021 post-monsoon and monsoon-onset rasters may have slightly "
              "different extents/resolutions (different source pipelines) — this figure "
              "is a visual qualitative comparison, NOT a pixel-aligned change-detection "
              "matrix like Paper 1's Module 8. Treat area percentages as independently "
              "computed per-raster summaries, not a validated transition matrix.")
    else:
        print(f"\nWARNING: {LULC_2021_PATH} not found, skipping post-monsoon comparison figure.")

    print("\n" + "=" * 60)
    print("  MODULE 11 COMPLETE")
    print(f"  Threshold reproduction check: {match_rate*100:.2f}% (expect 100.00%)")
    print(f"  Outputs: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()