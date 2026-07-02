"""
05_flood_baseline_median.py
============================
Extract a multi-year seasonal SAR median (VV, VH) and per-pixel standard deviation
over the Mahanadi AOI for June-September 2021, 2022, 2023.

This replaces the single-date July 2020 baseline that failed due to scene-level
wind/roughness variation between dates (ocean VV mean was -12.92 dB on Aug 26 --
too rough to threshold reliably). A per-pixel seasonal median cancels out
scene-level roughness offsets and gives a stable local reference.

Output: one GeoTIFF with 4 bands:
  Band 1: VV median  (dB, Float32)
  Band 2: VH median  (dB, Float32)
  Band 3: VV std     (dB, Float32)  -- used for z-score anomaly detection
  Band 4: VH std     (dB, Float32)

Filename: mahanadi_sar_seasonal_median_2021_2023.tif

Locked conventions:
  - AOI: Mahanadi [86.45, 20.15, 86.75, 20.45]
  - CRS: EPSG:32645
  - Orbit: DESCENDING only, single relative orbit (see below)
  - Seasons: June 1 - September 30 for 2021, 2022, 2023
  - Instrument mode: IW, VV+VH
  - Export: dimensions only (no scale), same 4096x4096 grid as flood SAR

RELATIVE ORBIT FILTER (bug fix, found after first threshold_check run):
  'DESCENDING' is only the pass DIRECTION, not a specific orbit track.
  Sentinel-1 typically covers a given AOI via multiple distinct descending
  relative orbits, each with a different incidence angle / look geometry.
  The first version of this script pooled scenes across ALL descending
  relative orbits into one per-pixel median/std. The Aug 26 2020 flood
  scene (script 04) is a SINGLE relative orbit. Comparing a single-orbit
  flood image against a median blended across multiple orbit geometries
  introduces a systematic incidence-angle-dependent offset, especially
  over agricultural/upland land cover (volume scattering from crops is
  far more incidence-angle sensitive than open water). This showed up as
  a z-score anomaly mask lighting up broadly across the upland interior
  instead of concentrating near the river channel/floodplain.

  Fix: filter this script's collection to the SAME relativeOrbitNumber_start
  as the flood scene, so every pixel's seasonal median/std is geometrically
  comparable to the Aug 26 flood image. Run check_flood_orbit.py first to
  get this number -- do not guess it.

Usage:
  python gee_scripts\\05_flood_baseline_median.py
"""

import ee

import argparse

# ── initialise GEE ────────────────────────────────────────────────────────────

ee.Initialize(project='sar-optical-synthesis')

# ── locked parameters ─────────────────────────────────────────────────────────

AOI = ee.Geometry.Rectangle([86.45, 20.15, 86.75, 20.45])

SEASON_WINDOWS = [
    ('2021-06-01', '2021-09-30'),
    ('2022-06-01', '2022-09-30'),
    ('2023-06-01', '2023-09-30'),
]

# Relative orbit number of the Aug 26 2020 flood scene (script 04), obtained
# by running check_flood_orbit.py FIRST. Do not guess this value -- if it is
# left as None, this script will refuse to run (see assertion in main()).
# TODO: fill in with the integer printed by check_flood_orbit.py
RELATIVE_ORBIT_NUMBER = 48

EXPORT_FOLDER     = 'SAR_Optical_FloodValidation'
EXPORT_FILENAME   = 'mahanadi_sar_seasonal_median_2021_2023'
CRS               = 'EPSG:32645'
DIMENSIONS        = '4096x4096'   # matches flood SAR export dimensions
GEE_PROJECT       = 'sar-optical-synthesis'


def get_s1_collection(start_date, end_date):
    """
    Return filtered S1 GRD IW descending collection over Mahanadi AOI,
    restricted to RELATIVE_ORBIT_NUMBER so every scene shares the same
    incidence-angle/look geometry as the Aug 26 2020 flood scene.
    """
    return (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(AOI)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
        .filter(ee.Filter.eq('relativeOrbitNumber_start', RELATIVE_ORBIT_NUMBER))
        .select(['VV', 'VH'])
    )


def main():
    if RELATIVE_ORBIT_NUMBER is None:
        raise RuntimeError(
            "RELATIVE_ORBIT_NUMBER is not set. Run "
            "`python gee_scripts\\check_flood_orbit.py` first, note the printed "
            "relativeOrbitNumber_start for the Aug 26 2020 flood scene, and set "
            "RELATIVE_ORBIT_NUMBER at the top of this script to that integer. "
            "Refusing to guess (same failure mode as the original GEE silent-skip "
            "bug documented in module1_build_manifest.py)."
        )

    print("Building multi-year seasonal SAR median for Mahanadi AOI...")
    print(f"  Seasons: {[w[0] + ' to ' + w[1] for w in SEASON_WINDOWS]}")
    print(f"  Orbit: DESCENDING, relativeOrbitNumber_start={RELATIVE_ORBIT_NUMBER} "
          f"| CRS: {CRS} | Dimensions: {DIMENSIONS}")

    # ── collect all scenes across the three seasons ───────────────────────────
    collections = [get_s1_collection(s, e) for s, e in SEASON_WINDOWS]

    # Print scene counts per season for verification
    for (s, e), col in zip(SEASON_WINDOWS, collections):
        count = col.size().getInfo()
        print(f"  {s} to {e}: {count} scenes")

    # Merge all three seasons into one collection
    all_scenes = collections[0]
    for col in collections[1:]:
        all_scenes = all_scenes.merge(col)

    total = all_scenes.size().getInfo()
    print(f"  Total scenes across all seasons: {total}")

    if total < 3:
        raise RuntimeError(
            f"Only {total} scenes found across 3 seasons with "
            f"relativeOrbitNumber_start={RELATIVE_ORBIT_NUMBER} -- expected at "
            "least 3-9 (roughly one matching pass per season per year, since "
            "restricting to a single relative orbit reduces revisit frequency "
            "compared to pooling all descending orbits). If this is unexpectedly "
            "low or zero, double check the orbit number against check_flood_orbit.py "
            "output, and confirm the AOI is fully within that orbit's swath for "
            "all three years."
        )

    # ── compute per-pixel median and std ─────────────────────────────────────
    # reducers run per-pixel across the time stack
    vv_median = all_scenes.select('VV').median().rename('VV_median')
    vh_median = all_scenes.select('VH').median().rename('VH_median')
    vv_std    = all_scenes.select('VV').reduce(ee.Reducer.stdDev()).rename('VV_std')
    vh_std    = all_scenes.select('VH').reduce(ee.Reducer.stdDev()).rename('VH_std')

    # Stack all 4 bands into one image -- all Float64 from GEE reducers,
    # cast explicitly to Float32 before bundling (avoids type mismatch on export)
    median_image = (
        ee.Image.cat([
            vv_median.toFloat(),
            vh_median.toFloat(),
            vv_std.toFloat(),
            vh_std.toFloat(),
        ])
        .clip(AOI)
    )

    print(f"\n  Bands in export image: VV_median, VH_median, VV_std, VH_std (all Float32)")

    # ── verify band names before export ──────────────────────────────────────
    band_names = median_image.bandNames().getInfo()
    print(f"  Confirmed band names: {band_names}")

    # ── submit export task ────────────────────────────────────────────────────
    task = ee.batch.Export.image.toDrive(
        image=median_image,
        description=EXPORT_FILENAME,
        folder=EXPORT_FOLDER,
        fileNamePrefix=EXPORT_FILENAME,
        dimensions=DIMENSIONS,          # no scale -- dimensions only (locked bug fix)
        crs=CRS,
        region=AOI,
        maxPixels=1e9,
        fileFormat='GeoTIFF',
    )

    task.start()
    print(f"\n  Export task submitted: {EXPORT_FILENAME}")
    print(f"  Task ID: {task.id}")
    print(f"  Destination: Google Drive / {EXPORT_FOLDER}/{EXPORT_FILENAME}.tif")
    print("\n  Monitor at: https://code.earthengine.google.com/tasks")
    print("  Once complete, download with:")
    print(f'    rclone copy gdrive:{EXPORT_FOLDER}/{EXPORT_FILENAME}.tif '
          r'E:\SAR-Optical-Synthesis\data\flood_validation\ --progress')


if __name__ == '__main__':
    main()