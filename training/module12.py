"""
MODULE 12 — DSAS Erosion Hotspot Overlay on Monsoon LULC
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
Overlays Paper 1's DSAS transect erosion classification on the
GAN-reconstructed monsoon LULC maps (module11 output), matching the
style of Paper 1's Module 8c (8c_erosion_hotspot_map.png) but for the
monsoon-season classification instead of post-monsoon 2024.

This is the LAST planned module for Paper 2's application section.
Per explicit decision: full Module 8 parity (pixel-aligned transition
matrix between monsoon years, monsoon-vs-post-monsoon change matrix)
was deliberately NOT built — Paper 1 and Paper 2 do not need to be
completely identical in scope, only methodologically consistent where
they overlap (MNDWI thresholds, coastal-strip mask, transect geometry,
classification scheme — all already shared). This overlay figure is
the one remaining piece judged worth adding.

INPUTS (already produced by module11.py and existing Paper 1 outputs):
    training/outputs_module11_lulc/lulc_monsoon_<year>_coastal_strip.tif
    data/paper1/outputs/dsas_transects_classified.csv

OUTPUTS:
    training/outputs_module12_hotspot_overlay/
        fig_lulc_monsoon_erosion_hotspots_<year>.png  (one per monsoon year)
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from shapely.geometry import Point

ROOT = r"E:\SAR-Optical-Synthesis"
LULC_DIR = os.path.join(ROOT, "training", "outputs_module11_lulc")
PAPER1_OUT_DIR = os.path.join(ROOT, "data", "paper1", "outputs")
DSAS_CSV = os.path.join(PAPER1_OUT_DIR, "dsas_transects_classified.csv")

OUTPUT_DIR = os.path.join(ROOT, "training", "outputs_module12_hotspot_overlay")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = {0: "Land", 1: "Intertidal", 2: "Water"}
CLASS_COLORS = {0: "#8B4513", 1: "#F4A460", 2: "#1E90FF"}
STUDY_AREA = "Brahmapur Coastline, Ganjam, Odisha"

YEAR_TO_DATE = {2021: "2021-06-03", 2022: "2022-06-10", 2023: "2023-06-05"}
YEARS = [2021, 2022, 2023]


def load_dsas_points():
    dsas_df = pd.read_csv(DSAS_CSV)
    dsas_gdf = gpd.GeoDataFrame(
        dsas_df,
        geometry=[Point(xy) for xy in zip(dsas_df["Longitude"], dsas_df["Latitude"])],
        crs="EPSG:4326",
    )
    # Transects are stored in lon/lat (EPSG:4326); LULC mosaics are in
    # EPSG:32645 (UTM 45N) — reproject for correct overlay placement.
    dsas_gdf_utm = dsas_gdf.to_crs(epsg=32645)
    return dsas_df, dsas_gdf_utm


def figure_hotspot_overlay(year, lulc_arr, lulc_profile, dsas_df, dsas_gdf_utm):
    high_erosion = dsas_gdf_utm[dsas_gdf_utm["Class"] == "High Erosion"]
    high_accretion = dsas_gdf_utm[dsas_gdf_utm["Class"] == "High Accretion"]
    stable = dsas_gdf_utm[dsas_gdf_utm["Class"] == "Stable"]

    cmap_lulc = ListedColormap([CLASS_COLORS[0], CLASS_COLORS[1], CLASS_COLORS[2]])
    norm_lulc = BoundaryNorm([0, 1, 2, 3], 3)

    bounds = rasterio.transform.array_bounds(
        lulc_arr.shape[0], lulc_arr.shape[1], lulc_profile["transform"]
    )
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    fig, ax = plt.subplots(figsize=(11, 13), facecolor="#F0F4F8")
    fig.suptitle(f"Monsoon LULC ({YEAR_TO_DATE[year]}) with DSAS Erosion Hotspots\n{STUDY_AREA}",
                 fontsize=13, fontweight="bold", y=0.99)

    masked = np.ma.masked_where(lulc_arr < 0, lulc_arr)  # -1 = outside coastal strip (nodata)
    ax.imshow(masked, cmap=cmap_lulc, norm=norm_lulc, extent=extent,
              aspect="auto", interpolation="nearest", alpha=0.85)

    scatter_cfg = [
        (high_erosion, "#FF0000", "High Erosion (LRR <= -1 m/yr)", 80, "v"),
        (high_accretion, "#00CC44", "High Accretion (LRR >= +1 m/yr)", 80, "^"),
        (stable, "#FFFF00", "Stable (-0.3 to +0.3 m/yr)", 40, "o"),
    ]
    for gdf, color, label, size, marker in scatter_cfg:
        if len(gdf) > 0:
            ax.scatter(gdf.geometry.x, gdf.geometry.y, c=color, s=size, label=label,
                       edgecolors="black", linewidth=0.4, zorder=5, alpha=0.9, marker=marker)

    ax.set_xlabel("Easting (m, UTM 45N)", fontsize=10)
    ax.set_ylabel("Northing (m, UTM 45N)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

    lulc_patches = [mpatches.Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i], alpha=0.85)
                     for i in range(3)]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=lulc_patches + handles, labels=list(CLASS_NAMES.values()) + labels,
              loc="upper right", fontsize=8.5, framealpha=0.9,
              title="LULC + DSAS Classes", title_fontsize=9)

    n_he, n_ha, n_st = len(high_erosion), len(high_accretion), len(stable)
    n_total = len(dsas_gdf_utm)
    stats_text = (
        f"DSAS Summary (n={n_total})\n"
        f"High Erosion   : {n_he} ({n_he/n_total*100:.0f}%)\n"
        f"High Accretion : {n_ha} ({n_ha/n_total*100:.0f}%)\n"
        f"Stable         : {n_st} ({n_st/n_total*100:.0f}%)\n"
        f"Mean LRR       : {dsas_df['LRR_m_yr'].mean():.3f} m/yr\n"
        f"(LULC: monsoon-onset {YEAR_TO_DATE[year]}, coastal strip only)"
    )
    ax.text(0.01, 0.01, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"fig_lulc_monsoon_erosion_hotspots_{year}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("=" * 60)
    print("  MODULE 12 — DSAS Erosion Hotspot Overlay on Monsoon LULC")
    print("  (final planned module for Paper 2's application section)")
    print("=" * 60)

    dsas_df, dsas_gdf_utm = load_dsas_points()
    print(f"DSAS transects loaded: {len(dsas_gdf_utm)}")

    for year in YEARS:
        lulc_path = os.path.join(LULC_DIR, f"lulc_monsoon_{year}_coastal_strip.tif")
        if not os.path.exists(lulc_path):
            print(f"  WARNING: {lulc_path} not found — run module11 first. Skipping {year}.")
            continue

        with rasterio.open(lulc_path) as src:
            lulc_arr = src.read(1)
            lulc_profile = src.profile.copy()

        figure_hotspot_overlay(year, lulc_arr, lulc_profile, dsas_df, dsas_gdf_utm)

    print("\n" + "=" * 60)
    print("  MODULE 12 COMPLETE")
    print(f"  Outputs: {OUTPUT_DIR}")
    print("  This is the final analysis module for Paper 2's application section.")
    print("=" * 60)


if __name__ == "__main__":
    main()