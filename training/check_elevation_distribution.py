"""
check_std_by_elevation.py
===========================
Checks whether VV_seasonal_std is anomalously LOW in the 1.4-4m elevation band
where flagged pixels concentrate. If std is unusually low there (e.g. due to a
land-cover type with very stable/uniform backscatter across the 2021-2023
seasonal window), that would mechanically inflate z-scores for any modest
absolute deviation on the flood date -- producing false positives that have
nothing to do with real flooding.

Also checks the raw (flood_VV - VV_median) numerator by elevation band, to see
whether the absolute backscatter DROP is itself elevation-correlated (which
would be more consistent with real partial inundation / waterlogging) or
whether it's the denominator (std) driving the effect (more consistent with
an artifact).

Requires:
  - mahanadi_sar_seasonal_median_2021_2023.tif  (4-band: VV_median, VH_median,
    VV_std, VH_std)
  - mahanadi_flood2020_flooddate_sar_2020-08-26.tif
  - mahanadi_dem_glo30.tif

Usage:
  python training\\check_std_by_elevation.py
"""

import os
import numpy as np
import rasterio

BASE_DIR   = r"E:\SAR-Optical-Synthesis"
FLOOD_DIR  = os.path.join(BASE_DIR, "data", "flood_validation")

SEASONAL_PATH = os.path.join(FLOOD_DIR, "mahanadi_sar_seasonal_median_2021_2023.tif")
FLOOD_PATH    = os.path.join(FLOOD_DIR, "mahanadi_flood2020_flooddate_sar_2020-08-26.tif")
DEM_PATH      = os.path.join(FLOOD_DIR, "mahanadi_dem_glo30.tif")

# Elevation bands to inspect (metres). The 1.4-4.2m band is where flagged
# z-score pixels concentrated (p5-p95 from the elevation distribution check).
ELEV_BANDS = [
    (-10, 0,   "river/lowest (<=0m)"),
    (0,   1.4, "low floodplain (0-1.4m)"),
    (1.4, 4.2, "SUSPECT BAND (1.4-4.2m, where flags concentrate)"),
    (4.2, 8,   "higher ground (4.2-8m)"),
    (8,   100, "highest (>8m)"),
]


def load_band(path, band_idx, label):
    with rasterio.open(path) as src:
        arr = src.read(band_idx).astype(np.float32)
    print(f"  Loaded {label}: band {band_idx} from {os.path.basename(path)}")
    return arr


def main():
    print("="*70)
    print("Checking VV_std and raw backscatter drop by elevation band")
    print("="*70)

    print("\n[1] Loading rasters...")
    vv_median = load_band(SEASONAL_PATH, 1, "VV_median")
    vv_std    = load_band(SEASONAL_PATH, 3, "VV_std")
    vv_flood  = load_band(FLOOD_PATH,    1, "flood VV")

    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1).astype(np.float32)

    shapes = {vv_median.shape, vv_std.shape, vv_flood.shape, dem.shape}
    if len(shapes) > 1:
        raise ValueError(f"Shape mismatch across rasters: {shapes}")

    valid = (
        np.isfinite(vv_median) & np.isfinite(vv_std) & np.isfinite(vv_flood) &
        np.isfinite(dem) & (dem > -100) & (dem < 1000) & (vv_std > 1e-6)
    )
    print(f"  Valid pixels overall: {valid.sum():,} / {dem.size:,}")

    raw_drop = vv_flood - vv_median   # negative = darker than seasonal norm
    z = raw_drop / np.where(vv_std > 1e-6, vv_std, 1e-6)

    print("\n[2] Stats by elevation band...")
    print(f"\n  {'Band':<42} {'n':>10} {'VV_std mean':>12} {'VV_std med':>11} "
          f"{'raw_drop mean':>14} {'z mean':>8} {'%flagged(z<-1.5)':>17}")
    print("  " + "-"*108)

    for lo, hi, label in ELEV_BANDS:
        band_mask = valid & (dem > lo) & (dem <= hi)
        n = band_mask.sum()
        if n == 0:
            print(f"  {label:<42} {0:>10}  (no pixels in this band)")
            continue
        std_vals  = vv_std[band_mask]
        drop_vals = raw_drop[band_mask]
        z_vals    = z[band_mask]
        pct_flagged = (z_vals < -1.5).mean() * 100

        print(f"  {label:<42} {n:>10,} {std_vals.mean():>12.3f} {np.median(std_vals):>11.3f} "
              f"{drop_vals.mean():>14.3f} {z_vals.mean():>8.3f} {pct_flagged:>16.1f}%")

    print("\n[3] Direct comparison: SUSPECT BAND vs REST OF AOI...")
    suspect = valid & (dem > 1.4) & (dem <= 4.2)
    rest    = valid & ~((dem > 1.4) & (dem <= 4.2))

    print(f"\n  SUSPECT BAND (1.4-4.2m, n={suspect.sum():,}):")
    print(f"    VV_std:    mean={vv_std[suspect].mean():.3f}  median={np.median(vv_std[suspect]):.3f}")
    print(f"    raw_drop:  mean={raw_drop[suspect].mean():.3f}  median={np.median(raw_drop[suspect]):.3f}")
    print(f"    z-score:   mean={z[suspect].mean():.3f}  median={np.median(z[suspect]):.3f}")

    print(f"\n  REST OF AOI (n={rest.sum():,}):")
    print(f"    VV_std:    mean={vv_std[rest].mean():.3f}  median={np.median(vv_std[rest]):.3f}")
    print(f"    raw_drop:  mean={raw_drop[rest].mean():.3f}  median={np.median(raw_drop[rest]):.3f}")
    print(f"    z-score:   mean={z[rest].mean():.3f}  median={np.median(z[rest]):.3f}")

    std_ratio = vv_std[suspect].mean() / vv_std[rest].mean()
    drop_ratio_diff = raw_drop[suspect].mean() - raw_drop[rest].mean()
    print(f"\n  VV_std ratio (suspect / rest): {std_ratio:.3f}")
    print(f"  raw_drop difference (suspect - rest): {drop_ratio_diff:.3f} dB")

    print("\n" + "="*70)
    print("INTERPRETATION GUIDE:")
    print("  - If VV_std is notably LOWER in the suspect band than the rest of the")
    print("    AOI, while raw_drop (the actual dB change) is SIMILAR across bands,")
    print("    that means the inflated z-score there is a DENOMINATOR artifact --")
    print("    i.e. land cover with unusually stable/low-variance backscatter is")
    print("    mechanically tripping the threshold, not real extra darkening.")
    print("  - If raw_drop is ALSO notably more negative in the suspect band (i.e.")
    print("    backscatter genuinely dropped more there), that's more consistent")
    print("    with real partial inundation/waterlogging specific to that elevation")
    print("    band -- possibly flooded cropland, which would support the mask as")
    print("    at least partially real, even if the std-driven inflation is also")
    print("    present.")
    print("="*70)


if __name__ == '__main__':
    main()