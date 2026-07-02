"""
MODULE 10 — Monsoon Shoreline Demonstration (Paper 2 Application Section)
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
Uses the trained GAN (checkpoints_run2_unweighted/gan_generator_epoch100.pt,
LOCKED — do not retrain or substitute) to reconstruct cloud-free optical
imagery for the Brahmapur AOI during three monsoon-onset dates
(2021-06-03, 2022-06-10, 2023-06-05), extracts MNDWI shorelines from the
reconstructed mosaics, and computes seasonal (monsoon vs. post-monsoon)
shoreline displacement at Paper 1's 119 DSAS transects.

METHODOLOGY CONSISTENCY WITH PAPER 1 (do not change these constants —
they are reused verbatim from the Paper 1 notebook so the monsoon vs.
post-monsoon comparison is methodologically apples-to-apples):
    - coastal_strip polygon (clips out Tampara Lake / inland geometry)
    - TRANSECT_SPACING = 250 m, TRANSECT_LENGTH = 2000 m
    - MAX_VALID_DIST = 500 m (inland/pond-hit sanity filter)
    - MNDWI shoreline threshold = 0.0

PIPELINE STEPS
--------------
1. Per-patch GAN inference (72 Brahmapur patches x 3 years)
2. Mosaic reconstructed patches into one 6-band raster per year
3. Compute MNDWI from mosaicked B3/B11 (module6 formula, reused)
4. Extract shoreline (threshold=0.0) within the coastal-strip mask
5. Regenerate Paper 1's 119 transects from shoreline_2013.shp
   (identical cast_transects() logic — deterministic, reproducible)
6. Compute signed projected distance at each transect, same formula
   as Paper 1: proj_d - (t_len / 2)
7. Compare against dist_2021/2022/2023 already in
   annual_distances_processed.csv (no need to re-derive post-monsoon
   side from shapefiles)
8. Figures + tables matching Paper 1 style

USAGE
-----
    python module10_monsoon_shoreline.py

All paths are configured in the CONFIG block below — edit those, not
the logic beneath them.
"""

import os
import math
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.merge import merge as rasterio_merge
from rasterio.transform import Affine
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from shapely.ops import unary_union, linemerge
from shapely.geometry import (
    MultiPolygon, Polygon, MultiLineString, LineString, Point, box
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these paths if your layout differs
# ─────────────────────────────────────────────────────────────────────────────
ROOT = r"E:\SAR-Optical-Synthesis"

SAR_DIR        = os.path.join(ROOT, "data", "sar_gap_input")
REF_DIR        = os.path.join(ROOT, "data", "optical_reference")
MANIFEST_CSV   = os.path.join(ROOT, "data", "manifest.csv")
CHECKPOINT     = os.path.join(ROOT, "checkpoints_run2_unweighted", "gan_generator_epoch100.pt")

PAPER1_DIR     = os.path.join(ROOT, "data", "paper1")
PAPER1_OUT_DIR = os.path.join(PAPER1_DIR, "outputs")
SHORELINE_2013 = os.path.join(PAPER1_DIR, "shoreline_2013.shp")
DSAS_CSV       = os.path.join(PAPER1_OUT_DIR, "dsas_transects_classified.csv")
ANNUAL_DIST_CSV = os.path.join(PAPER1_OUT_DIR, "annual_distances_processed.csv")

OUTPUT_DIR = os.path.join(ROOT, "training", "outputs_module10")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Allow importing module4 (AttentionUNet) and module2 (normalization fns)
# from the training/ directory regardless of cwd.
TRAINING_DIR = os.path.join(ROOT, "training")
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)

from module4_attention_unet import AttentionUNet          # noqa: E402
from module2_dataset import normalize_sar, normalize_optical, read_tif_as_array  # noqa: E402
from module6_evaluation import compute_mndwi               # noqa: E402

AOI_NAME   = "brahmapur"
PATCH_IDS  = [f"{i:04d}" for i in range(72)]   # 0000-0071, confirmed on disk
YEAR_TO_DATE = {
    2021: "2021-06-03",
    2022: "2022-06-10",
    2023: "2023-06-05",
}
REF_DATE_FOR_YEAR = {
    2021: "2021-06-01",
    2022: "2022-06-01",
    2023: "2023-06-01",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# LOCKED CONSTANTS — reused VERBATIM from the Paper 1 notebook.
# Do not change. Methodological consistency between Paper 1 and Paper 2
# depends on these being identical.
# ─────────────────────────────────────────────────────────────────────────────
TRANSECT_SPACING = 250    # metres
TRANSECT_LENGTH  = 2000   # metres
MAX_VALID_DIST   = 500    # metres — sanity filter for inland/pond hits
MNDWI_THRESHOLD  = 0.0

COASTAL_STRIP_COORDS = [
    (85.015, 19.400),   # top-left
    (85.042, 19.400),   # top-right
    (84.878, 19.200),   # bottom-right
    (84.840, 19.200),   # bottom-left
]

OPTICAL_BAND_ORDER = ["B2", "B3", "B4", "B8", "B11", "B12"]
B3_IDX, B11_IDX = OPTICAL_BAND_ORDER.index("B3"), OPTICAL_BAND_ORDER.index("B11")
B4_IDX, B2_IDX = OPTICAL_BAND_ORDER.index("B4"), OPTICAL_BAND_ORDER.index("B2")


# =============================================================================
# STEP 1 — GAN INFERENCE PER PATCH
# =============================================================================

def load_generator(checkpoint_path):
    model = AttentionUNet(in_channels=9, out_channels=6, base_ch=64).to(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded generator checkpoint: {checkpoint_path}")
    return model


def build_input_tensor(sar_path, ref_path):
    """Replicates module2_dataset.SAROpticalTripletDataset.__getitem__
    input-assembly logic exactly (normalization functions imported
    directly from module2, not reimplemented)."""
    sar = read_tif_as_array(sar_path, expected_bands=2)
    ref = read_tif_as_array(ref_path, expected_bands=7)
    ref_optical = ref[:6]
    ref_cloud_mask = ref[6:7]

    sar_norm = normalize_sar(np.nan_to_num(sar, nan=0.0))
    ref_norm = normalize_optical(ref_optical)

    input_stack = np.concatenate([sar_norm, ref_norm, ref_cloud_mask], axis=0)  # (9,256,256)
    return torch.from_numpy(input_stack).unsqueeze(0).float()  # (1,9,256,256)


def reconstruct_year(generator, year):
    """Runs GAN inference on all 72 Brahmapur patches for one monsoon year.
    Returns list of (numpy 6-band array [0,1], rasterio profile) tuples,
    each carrying the SOURCE PATCH's own georeferencing — needed for
    rasterio.merge in step 2."""
    sar_date = YEAR_TO_DATE[year]
    ref_date = REF_DATE_FOR_YEAR[year]

    reconstructed = []
    missing = []

    for pid in PATCH_IDS:
        sar_path = os.path.join(SAR_DIR, f"s1_{AOI_NAME}_{pid}_{sar_date}.tif")
        ref_path = os.path.join(REF_DIR, f"s2ref_{AOI_NAME}_{pid}_{ref_date}.tif")

        if not (os.path.exists(sar_path) and os.path.exists(ref_path)):
            missing.append(pid)
            continue

        with rasterio.open(sar_path) as src:
            profile = src.profile.copy()

        input_tensor = build_input_tensor(sar_path, ref_path).to(DEVICE)
        with torch.no_grad():
            pred = generator(input_tensor)  # (1,6,256,256), sigmoid output in [0,1]
        pred_np = pred.squeeze(0).cpu().numpy().astype(np.float32)  # (6,256,256)

        profile.update(count=6, dtype="float32")
        reconstructed.append((pred_np, profile, pid))

    if missing:
        print(f"  WARNING [{year}]: {len(missing)} patches missing on disk: {missing}")

    print(f"Year {year}: reconstructed {len(reconstructed)}/{len(PATCH_IDS)} patches")
    return reconstructed


# =============================================================================
# STEP 2 — MOSAIC RECONSTRUCTED PATCHES INTO ONE AOI RASTER PER YEAR
# =============================================================================

def mosaic_year(reconstructed_patches, year):
    """Writes each patch to a temp in-memory dataset, then rasterio.merge
    into a single 6-band mosaic. Patches already carry correct embedded
    affine transforms (confirmed via rasterio inspection), so no external
    patch-grid file is required."""
    memfiles = []
    datasets = []
    try:
        for pred_np, profile, pid in reconstructed_patches:
            memfile = rasterio.io.MemoryFile()
            with memfile.open(**profile) as dst:
                dst.write(pred_np)
            memfiles.append(memfile)
            datasets.append(memfile.open())

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

    out_path = os.path.join(OUTPUT_DIR, f"reconstructed_optical_mosaic_{year}.tif")
    with rasterio.open(out_path, "w", **mosaic_profile) as dst:
        dst.write(mosaic_arr)
    print(f"Year {year}: mosaic saved -> {out_path}  shape={mosaic_arr.shape}")

    return mosaic_arr, mosaic_profile


# =============================================================================
# STEP 3-4 — MNDWI + SHORELINE EXTRACTION WITHIN COASTAL STRIP MASK
# =============================================================================

def extract_mndwi_shoreline(mosaic_arr, mosaic_profile, year):
    """mosaic_arr: (6, H, W) float32 [0,1]. Returns:
        mndwi (H,W) array, shoreline_gdf (GeoDataFrame of LineStrings in
        EPSG:32645, clipped to the coastal strip)."""
    green = mosaic_arr[B3_IDX]
    swir1 = mosaic_arr[B11_IDX]
    mndwi = compute_mndwi(np.stack([mosaic_arr[i] for i in range(6)], axis=0))

    # Save MNDWI raster for figure generation
    mndwi_path = os.path.join(OUTPUT_DIR, f"mndwi_{year}.tif")
    mndwi_profile = mosaic_profile.copy()
    mndwi_profile.update(count=1, dtype="float32")
    with rasterio.open(mndwi_path, "w", **mndwi_profile) as dst:
        dst.write(mndwi[np.newaxis, :, :])

    # Binary water mask at the LOCKED threshold
    water_mask = (mndwi > MNDWI_THRESHOLD).astype(np.uint8)

    # Vectorize the land/water boundary
    from rasterio.features import shapes as rio_shapes
    transform = mosaic_profile["transform"]
    polygons = []
    for geom, val in rio_shapes(water_mask, transform=transform):
        if val == 1:
            polygons.append(Polygon(geom["coordinates"][0]))

    if len(polygons) == 0:
        raise RuntimeError(f"Year {year}: no water polygons found — check MNDWI threshold/mosaic.")

    water_union = unary_union(polygons)
    boundary = water_union.boundary  # LineString or MultiLineString

    shoreline_gdf = gpd.GeoDataFrame(geometry=[boundary], crs="EPSG:32645")

    # Clip to the SAME coastal strip mask used in Paper 1 (locked constants)
    coastal_strip = Polygon(COASTAL_STRIP_COORDS)
    coastal_strip_gdf = gpd.GeoDataFrame(geometry=[coastal_strip], crs="EPSG:4326").to_crs(epsg=32645)
    shoreline_clipped = gpd.clip(shoreline_gdf, coastal_strip_gdf)

    return mndwi, shoreline_clipped


# =============================================================================
# STEP 5 — REGENERATE PAPER 1'S 119 TRANSECTS (identical logic to the notebook)
# =============================================================================

def extract_lines(geom):
    if isinstance(geom, Polygon):
        return geom.exterior
    elif isinstance(geom, MultiPolygon):
        return linemerge([p.exterior for p in geom.geoms])
    elif isinstance(geom, MultiLineString):
        return linemerge(geom)
    elif isinstance(geom, LineString):
        return geom
    else:
        return geom.boundary


def cast_transects(baseline_geom, spacing, length):
    """Verbatim reproduction of the Paper 1 notebook's cast_transects().
    Deterministic given the same baseline + spacing + length, so
    TransectID alignment with dsas_transects_classified.csv is preserved."""
    if isinstance(baseline_geom, MultiLineString):
        lines = list(baseline_geom.geoms)
    else:
        lines = [baseline_geom]

    transects = []
    tid = 0
    for line in lines:
        total_len = line.length
        d = 0.0
        while d <= total_len:
            pt = line.interpolate(d)
            d2 = min(d + 1.0, total_len)
            pt2 = line.interpolate(d2)
            dx = pt2.x - pt.x
            dy = pt2.y - pt.y
            norm = math.hypot(dx, dy)
            if norm == 0:
                d += spacing
                continue
            px, py = -dy / norm, dx / norm
            half = length / 2
            start = Point(pt.x - px * half, pt.y - py * half)
            end = Point(pt.x + px * half, pt.y + py * half)
            transect_line = LineString([start, end])

            midpt_gdf = gpd.GeoDataFrame(geometry=[pt], crs="EPSG:32645").to_crs(epsg=4326)
            lon = midpt_gdf.geometry.x.values[0]
            lat = midpt_gdf.geometry.y.values[0]

            transects.append({
                "TransectID": tid,
                "Longitude": round(lon, 6),
                "Latitude": round(lat, 6),
                "geometry": transect_line,
            })
            tid += 1
            d += spacing

    return gpd.GeoDataFrame(transects, crs="EPSG:32645")


def regenerate_paper1_transects():
    baseline_raw = gpd.read_file(SHORELINE_2013)
    if baseline_raw.crs is None or baseline_raw.crs.to_epsg() != 32645:
        baseline_raw = baseline_raw.to_crs(epsg=32645)

    coastal_strip = Polygon(COASTAL_STRIP_COORDS)
    coastal_strip_gdf = gpd.GeoDataFrame(geometry=[coastal_strip], crs="EPSG:4326").to_crs(epsg=32645)
    baseline_clipped = gpd.clip(baseline_raw, coastal_strip_gdf)

    if len(baseline_clipped) == 0:
        raise ValueError("Coastal corridor clip returned empty geometry — check shoreline_2013.shp.")

    baseline_union = unary_union(baseline_clipped.geometry.values)
    baseline_union = extract_lines(baseline_union)

    transects_gdf = cast_transects(baseline_union, spacing=TRANSECT_SPACING, length=TRANSECT_LENGTH)
    print(f"Regenerated {len(transects_gdf)} transects (pre-filter, matches Paper 1 raw count)")
    return transects_gdf


# =============================================================================
# STEP 6 — SIGNED PROJECTED DISTANCE AT EACH TRANSECT (identical formula)
# =============================================================================

def compute_signed_distances(transects_gdf, shoreline_line):
    """Identical signing convention to Paper 1:
        signed = proj_d - (t_len / 2)
        negative = landward of baseline (erosion), positive = seaward.
    Applies the SAME MAX_VALID_DIST sanity filter."""
    dists = []
    for _, t_row in transects_gdf.iterrows():
        try:
            intersection = t_row.geometry.intersection(shoreline_line)
            if intersection.is_empty:
                dists.append(np.nan)
                continue

            if intersection.geom_type == "MultiPoint":
                int_pt = list(intersection.geoms)[0]
            elif intersection.geom_type == "Point":
                int_pt = intersection
            else:
                int_pt = intersection.centroid

            t_line = t_row.geometry
            t_len = t_line.length
            proj_d = t_line.project(int_pt)
            signed = proj_d - (t_len / 2)

            if abs(signed) > MAX_VALID_DIST:
                dists.append(np.nan)
            else:
                dists.append(round(signed, 2))
        except Exception:
            dists.append(np.nan)

    return np.array(dists)


# =============================================================================
# STEP 7 — DISPLACEMENT TABLE: MONSOON vs. POST-MONSOON AT MATCHED TRANSECTS
# =============================================================================

def build_displacement_table(transects_gdf, monsoon_shorelines):
    """monsoon_shorelines: dict {year: shapely shoreline line geometry}"""
    dsas_df = pd.read_csv(DSAS_CSV)
    annual_df = pd.read_csv(ANNUAL_DIST_CSV)

    # Align transects_gdf to the SAME surviving TransectIDs as the locked
    # Paper 1 outputs (dsas_df was filtered: dropna(LRR) + |LRR|<8 range).
    valid_ids = set(dsas_df["TransectID"].astype(int))
    transects_gdf = transects_gdf[transects_gdf["TransectID"].isin(valid_ids)].reset_index(drop=True)
    print(f"Aligned to {len(transects_gdf)} surviving Paper 1 transects "
          f"(expected 119; if different, check baseline/coastal-strip drift)")

    rows = []
    for year, shoreline_line in monsoon_shorelines.items():
        monsoon_dist = compute_signed_distances(transects_gdf, shoreline_line)
        postmonsoon_col = f"dist_{year}"

        for i, tid in enumerate(transects_gdf["TransectID"].values):
            pm_row = annual_df.loc[annual_df["TransectID"] == tid, postmonsoon_col]
            pm_dist = pm_row.values[0] if len(pm_row) else np.nan
            cls_row = dsas_df.loc[dsas_df["TransectID"] == tid, "Class"]
            cls = cls_row.values[0] if len(cls_row) else "Unknown"
            lrr_row = dsas_df.loc[dsas_df["TransectID"] == tid, "LRR_m_yr"]
            lrr = lrr_row.values[0] if len(lrr_row) else np.nan

            rows.append({
                "TransectID": tid,
                "Year": year,
                "Monsoon_dist_m": monsoon_dist[i],
                "PostMonsoon_dist_m": pm_dist,
                "Seasonal_displacement_m": (
                    monsoon_dist[i] - pm_dist
                    if not (np.isnan(monsoon_dist[i]) or np.isnan(pm_dist))
                    else np.nan
                ),
                "Class": cls,
                "LRR_m_yr": lrr,
            })

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "seasonal_displacement_table.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df


def summarize_displacement(df):
    print("\n" + "=" * 60)
    print("  MODULE 10 SUMMARY — Monsoon Seasonal Displacement")
    print("=" * 60)

    valid = df.dropna(subset=["Seasonal_displacement_m"])
    n_excluded = len(df) - len(valid)
    print(f"Valid transect-year observations: {len(valid)} / {len(df)} "
          f"({n_excluded} excluded by MAX_VALID_DIST={MAX_VALID_DIST}m filter)")

    by_year = valid.groupby("Year")["Seasonal_displacement_m"].agg(["mean", "std", "count"])
    print("\nMean seasonal displacement per year (m, monsoon - post-monsoon):")
    print(by_year.to_string())

    by_class = valid.groupby("Class")["Seasonal_displacement_m"].agg(
        mean_displacement="mean", std_displacement="std", n="count"
    )
    print("\nSeasonal displacement variability by Paper 1 erosion class:")
    print(by_class.to_string())

    pct_eroding_monsoon = (valid["Monsoon_dist_m"] < 0).mean() * 100
    pct_eroding_postmonsoon = (valid["PostMonsoon_dist_m"] < 0).mean() * 100
    print(f"\n% transects landward-of-baseline (monsoon): {pct_eroding_monsoon:.1f}%")
    print(f"% transects landward-of-baseline (post-monsoon): {pct_eroding_postmonsoon:.1f}%")

    summary_path = os.path.join(OUTPUT_DIR, "seasonal_displacement_summary.csv")
    by_year.to_csv(summary_path)
    print(f"\nSaved: {summary_path}")

    return by_year, by_class


# =============================================================================
# STEP 8 — FIGURES (matching Paper 1 style)
# =============================================================================

def figure_reconstructed_rgb(mosaics, years):
    fig, axes = plt.subplots(1, len(years), figsize=(6 * len(years), 6))
    if len(years) == 1:
        axes = [axes]
    for ax, year in zip(axes, years):
        mosaic_arr, _ = mosaics[year]
        rgb = np.stack([mosaic_arr[B4_IDX], mosaic_arr[1], mosaic_arr[B2_IDX]], axis=-1)  # B4,B3,B2
        rgb = np.clip(rgb * 3.5, 0, 1)
        ax.imshow(rgb)
        ax.set_title(f"Reconstructed optical — {YEAR_TO_DATE[year]}")
        ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_reconstructed_rgb.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def figure_mndwi_with_shoreline(mndwi_dict, shoreline_dict, years):
    fig, axes = plt.subplots(1, len(years), figsize=(6 * len(years), 6))
    if len(years) == 1:
        axes = [axes]
    for ax, year in zip(axes, years):
        ax.imshow(mndwi_dict[year], cmap="RdYlBu", vmin=-1, vmax=1)
        if not shoreline_dict[year].empty:
            shoreline_dict[year].plot(ax=ax, color="black", linewidth=1.2)
        ax.set_title(f"MNDWI + shoreline — {year} monsoon")
        ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_mndwi_shoreline.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def figure_displacement_bar(df):
    valid = df.dropna(subset=["Seasonal_displacement_m"])
    fig, ax = plt.subplots(figsize=(14, 5))

    CLASS_COLORS = {
        "High Erosion": "#D62728",
        "Moderate Erosion": "#FF7F0E",
        "Stable": "#2CA02C",
        "Moderate Accretion": "#1F77B4",
        "High Accretion": "#0D4F8B",
    }
    colors = valid["Class"].map(CLASS_COLORS).fillna("#999999")

    mean_by_transect = valid.groupby("TransectID").agg(
        Seasonal_displacement_m=("Seasonal_displacement_m", "mean"),
        Class=("Class", "first"),
    ).reset_index()
    bar_colors = mean_by_transect["Class"].map(CLASS_COLORS).fillna("#999999")

    ax.bar(mean_by_transect["TransectID"], mean_by_transect["Seasonal_displacement_m"],
           color=bar_colors, width=1.0)
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Transect ID (South -> North)")
    ax.set_ylabel("Mean seasonal displacement (m)\n(monsoon - post-monsoon)")
    ax.set_title("Seasonal Shoreline Displacement at Paper 1 Transects\nBrahmapur Coastline")

    legend_patches = [mpatches.Patch(facecolor=v, label=k) for k, v in CLASS_COLORS.items()]
    ax.legend(handles=legend_patches, fontsize=8.5, loc="upper right", framealpha=0.8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_seasonal_displacement_bar.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  MODULE 10 — Monsoon Shoreline Demonstration")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")

    generator = load_generator(CHECKPOINT)

    years = [2021, 2022, 2023]
    mosaics = {}
    mndwi_dict = {}
    shoreline_dict = {}

    for year in years:
        reconstructed = reconstruct_year(generator, year)
        mosaic_arr, mosaic_profile = mosaic_year(reconstructed, year)
        mosaics[year] = (mosaic_arr, mosaic_profile)

        mndwi, shoreline_gdf = extract_mndwi_shoreline(mosaic_arr, mosaic_profile, year)
        mndwi_dict[year] = mndwi
        shoreline_dict[year] = shoreline_gdf

    transects_gdf = regenerate_paper1_transects()

    monsoon_shoreline_geoms = {
        year: unary_union(shoreline_dict[year].geometry.values)
        for year in years
        if not shoreline_dict[year].empty
    }

    displacement_df = build_displacement_table(transects_gdf, monsoon_shoreline_geoms)
    by_year, by_class = summarize_displacement(displacement_df)

    figure_reconstructed_rgb(mosaics, years)
    figure_mndwi_with_shoreline(mndwi_dict, shoreline_dict, years)
    figure_displacement_bar(displacement_df)

    print("\n" + "=" * 60)
    print("  MODULE 10 COMPLETE")
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()