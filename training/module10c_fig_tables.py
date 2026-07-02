"""
MODULE 10c — GAN vs. Ground-Truth Validation Figure & Table
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
PURPOSE: Produces the validation figure/table for Paper 2's application
section, supporting the claim:

    "GAN-reconstructed monsoon shorelines agree closely with available
    ground-truth optical data (where ground truth exists), validating
    the reconstruction as faithful rather than artifactual — enabling
    the first monsoon-season shoreline record for this coastline."

This does NOT claim the GAN is "more accurate than Paper 1" — Paper 1
has no monsoon-period shoreline data to compare against (that is
precisely the gap this paper fills). The claim is narrower and
defensible: GAN output matches real ground truth where ground truth
exists (2021, 2023 — confirmed has_gt=True), so the GAN-only years
(2022, and any GT-missing patches in 2021/2023) can be trusted.

INPUTS (already produced by module10.py and module10b.py — this script
does not regenerate them, only compares):
    training/outputs_module10/seasonal_displacement_table.csv
    training/outputs_module10b_diagnostic/groundtruth_seasonal_displacement.csv

OUTPUTS:
    training/outputs_module10c_validation/
        fig_gan_vs_groundtruth_validation.png   (paper figure)
        table_gan_vs_groundtruth_validation.csv (paper table)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = r"E:\SAR-Optical-Synthesis"
GAN_CSV = os.path.join(ROOT, "training", "outputs_module10", "seasonal_displacement_table.csv")
GT_CSV = os.path.join(ROOT, "training", "outputs_module10b_diagnostic", "groundtruth_seasonal_displacement.csv")

OUTPUT_DIR = os.path.join(ROOT, "training", "outputs_module10c_validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Only years where ground truth actually exists — confirmed earlier
# (2022 has no GT patches, cannot be validated this way).
GT_VALIDATABLE_YEARS = [2021, 2023]


def load_data():
    gan_df = pd.read_csv(GAN_CSV)
    gt_df = pd.read_csv(GT_CSV)
    return gan_df, gt_df


def build_comparison_table(gan_df, gt_df):
    """Per-transect, per-year comparison of GAN monsoon distance vs.
    real ground-truth monsoon distance, for years where both exist."""
    rows = []

    gan_sub = gan_df[gan_df["Year"].isin(GT_VALIDATABLE_YEARS)].copy()
    gt_sub = gt_df[gt_df["Year"].isin(GT_VALIDATABLE_YEARS)].copy()

    merged = pd.merge(
        gan_sub[["TransectID", "Year", "Monsoon_dist_m", "Class", "LRR_m_yr"]],
        gt_sub[["TransectID", "Year", "GT_dist_m"]],
        on=["TransectID", "Year"],
        how="inner",
    )
    merged["GAN_minus_GT_m"] = merged["Monsoon_dist_m"] - merged["GT_dist_m"]
    merged["abs_error_m"] = merged["GAN_minus_GT_m"].abs()

    out_path = os.path.join(OUTPUT_DIR, "table_gan_vs_groundtruth_validation.csv")
    merged.to_csv(out_path, index=False)
    print(f"Saved per-transect validation table: {out_path}")
    print(f"  ({len(merged)} matched transect-year observations, both GAN and GT valid)")

    return merged


def summarize_validation(merged):
    valid = merged.dropna(subset=["GAN_minus_GT_m"])

    print("\n" + "=" * 60)
    print("  GAN vs GROUND-TRUTH VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Matched, valid observations: {len(valid)} / {len(merged)}")

    overall = {
        "Metric": "Mean absolute error (GAN vs GT shoreline distance)",
        "Value_m": round(valid["abs_error_m"].mean(), 2),
        "Std_m": round(valid["abs_error_m"].std(), 2),
        "N": len(valid),
    }
    print(f"\nMean absolute error: {overall['Value_m']} +/- {overall['Std_m']} m  (n={overall['N']})")

    corr = valid["Monsoon_dist_m"].corr(valid["GT_dist_m"])
    print(f"Correlation (GAN distance vs. GT distance): r = {corr:.3f}")

    pct_gan_landward = (valid["Monsoon_dist_m"] < 0).mean() * 100
    pct_gt_landward = (valid["GT_dist_m"] < 0).mean() * 100
    print(f"% landward — GAN: {pct_gan_landward:.1f}%   GT: {pct_gt_landward:.1f}%  "
          f"(difference: {abs(pct_gan_landward - pct_gt_landward):.1f} pp)")

    by_class = valid.groupby("Class").agg(
        mean_abs_error_m=("abs_error_m", "mean"),
        n=("abs_error_m", "count"),
    )
    print("\nMean absolute error by Paper 1 erosion class:")
    print(by_class.to_string())

    summary_rows = [
        {"Metric": "Mean absolute error (m)", "Value": round(valid["abs_error_m"].mean(), 2)},
        {"Metric": "Std of absolute error (m)", "Value": round(valid["abs_error_m"].std(), 2)},
        {"Metric": "Correlation (GAN vs GT distance), r", "Value": round(corr, 3)},
        {"Metric": "% landward — GAN", "Value": round(pct_gan_landward, 1)},
        {"Metric": "% landward — Ground truth", "Value": round(pct_gt_landward, 1)},
        {"Metric": "N matched observations", "Value": len(valid)},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "table_validation_summary_stats.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary stats table: {summary_path}")

    return valid, corr


def figure_validation(valid, corr):
    """Two-panel figure: (a) scatter GAN vs GT distance with 1:1 line,
    (b) per-transect bar comparison along the coastline."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A — scatter, GAN vs GT
    ax = axes[0]
    ax.scatter(valid["GT_dist_m"], valid["Monsoon_dist_m"], alpha=0.6, s=30,
               c="#1F77B4", edgecolors="white", linewidths=0.5)
    lims = [
        min(valid["GT_dist_m"].min(), valid["Monsoon_dist_m"].min()) - 20,
        max(valid["GT_dist_m"].max(), valid["Monsoon_dist_m"].max()) + 20,
    ]
    ax.plot(lims, lims, "k--", linewidth=1.2, label="1:1 line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Ground-truth shoreline distance (m)")
    ax.set_ylabel("GAN-reconstructed shoreline distance (m)")
    ax.set_title(f"GAN vs. Ground Truth\n(r = {corr:.3f}, n = {len(valid)})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel B — per-transect comparison along coastline, color by year
    ax2 = axes[1]
    YEAR_COLORS = {2021: "#2CA02C", 2023: "#D62728"}
    for year, color in YEAR_COLORS.items():
        sub = valid[valid["Year"] == year].sort_values("TransectID")
        if sub.empty:
            continue
        ax2.plot(sub["TransectID"], sub["GT_dist_m"], color=color, linewidth=1.5,
                 alpha=0.9, label=f"{year} Ground Truth")
        ax2.plot(sub["TransectID"], sub["Monsoon_dist_m"], color=color, linewidth=1.5,
                 linestyle="--", alpha=0.6, label=f"{year} GAN")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax2.set_xlabel("Transect ID (South -> North)")
    ax2.set_ylabel("Shoreline distance from baseline midpoint (m)")
    ax2.set_title("Per-Transect Agreement Along Coastline")
    ax2.legend(fontsize=8, ncol=2, loc="best")
    ax2.grid(alpha=0.3)

    plt.suptitle("Validation: GAN-Reconstructed vs. Real Ground-Truth Monsoon Shorelines\n"
                 "Brahmapur Coastline — 2021 & 2023 (years with available ground truth)",
                 fontsize=12, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "fig_gan_vs_groundtruth_validation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure: {out_path}")


def main():
    print("=" * 60)
    print("  MODULE 10c — GAN vs Ground-Truth Validation")
    print("=" * 60)

    gan_df, gt_df = load_data()
    merged = build_comparison_table(gan_df, gt_df)
    valid, corr = summarize_validation(merged)
    figure_validation(valid, corr)

    print("\n" + "=" * 60)
    print("  MODULE 10c COMPLETE")
    print(f"  Outputs: {OUTPUT_DIR}")
    print("=" * 60)
    print("""
  NOTE ON FRAMING for the paper text:
  This validates that GAN output matches real ground truth WHERE
  ground truth exists (2021, 2023). It does NOT claim the GAN is "more
  accurate than Paper 1" — Paper 1 has no monsoon-period data to
  compare against. The defensible claim is:
      "GAN-reconstructed monsoon shorelines agree closely with
      available ground-truth optical data, validating the
      reconstruction as faithful rather than artifactual, enabling
      the first monsoon-season shoreline record for this coastline."
  Use the mean absolute error / correlation numbers above as the
  quantitative backing for that sentence.
""")


if __name__ == "__main__":
    main()