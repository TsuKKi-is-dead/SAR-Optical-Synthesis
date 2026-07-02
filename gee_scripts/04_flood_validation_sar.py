"""
SCRIPT 04 — Flood Validation SAR Extraction (2020 Mahanadi flood)
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
Purpose: build an INDEPENDENT, self-derived SAR-threshold flood
reference mask for the August 2020 Mahanadi delta flood, to validate
the trained GAN's reconstructed-optical-derived NDWI/MNDWI against.

This is NOT part of the 762-patch training dataset. GAP_WINDOWS in
module1 only covers 2021-2023; the 2020 flood predates that entirely.
This script is a separate, one-off extraction for the flood-mapping
application section of Paper 2.

----------------------------------------------------------------
WHY WE BUILD OUR OWN REFERENCE MASK INSTEAD OF USING A THIRD-PARTY
FLOOD PRODUCT (Copernicus EMS / UNOSAT)
----------------------------------------------------------------
The Copernicus EMS rapid-mapping archive was checked directly (by the
user, via the EMS activations portal) for the Aug-Sep 2020 window:
activations existed for Andhra Pradesh and Assam, but NONE covering
the Odisha/Mahanadi delta flood specifically. There is no third-party
delineation product to validate against for this exact event.

This is also methodologically consistent with the paper's actual
argument (SAR penetrates cloud, optical doesn't) and avoids importing
someone else's sensor/threshold/date choices as an unexplained
external dependency. Note: EMS/UNOSAT-style maps are themselves
photo-interpreted "reference labels," not field-validated ground
truth (see WorldFloods, Scientific Reports 2021) -- so a carefully
derived SAR threshold is not epistemically weaker, just independently
sourced.

----------------------------------------------------------------
WHY A PRE-FLOOD BASELINE IS NEEDED (not just thresholding the flood date)
----------------------------------------------------------------
Mahanadi delta has substantial PERMANENT water (river channels,
distributaries, aquaculture ponds, paddy in standing water). A naive
single-date SAR water threshold would flag all of that as "flood,"
overstating flood extent. Standard practice (see GIS4Schools EMS
documentation) is to difference a flood-date mask against a pre-flood
baseline mask -- pixels that are water in BOTH dates are permanent
water; pixels that are water ONLY in the flood-date image are
flood-induced. This script extracts both dates so that differencing
can happen in module9 (the validation/comparison script, written
after this extraction completes).

----------------------------------------------------------------
AOI AND PATCH GRID -- REUSES EXISTING TRAINING ASSETS
----------------------------------------------------------------
Uses the SAME Mahanadi AOI and the SAME existing patch grid asset as
the 762-patch training dataset, NOT a new/different bounding box.
Rationale: the flood validation should cover the identical geography
the model was trained/tested on, so there's no question of having
selected a more favorable sub-region for validation. This also lets
the flood-date extraction reuse the existing patch tiling, so the
trained model's patch-based inference can run directly on the new
flood-date SAR without re-deriving a grid.

AOI:   Mahanadi = [86.45, 20.15, 86.75, 20.45]   (UNCHANGED from training)
Asset: users/mohitpradhan/mahanadi_patch_grid     (UNCHANGED from training)
CRS:   EPSG:32645 (UTM 45N)                       (UNCHANGED -- bug #2 fix)
Orbit: DESCENDING only                            (UNCHANGED -- locked decision #11)
Export: dimensions only, NOT scale (Export.image.toDrive cannot have
        both -- bug #1)

----------------------------------------------------------------
HOW TO USE
----------------------------------------------------------------
Step 1 (THIS SCRIPT, mode="list_dates"): list all available Sentinel-1
descending-orbit acquisition dates over the Mahanadi AOI in a search
window, for BOTH the flood period and a pre-flood candidate window.
Inspect the printed dates yourself before picking real ones --
Sentinel-1's 6-12 day revisit means you cannot just assume a date
exists; this avoids silently extracting the wrong/missing scene
(same class of bug as the original GEE silent-skip issue documented
in module1_build_manifest.py).

Step 2 (THIS SCRIPT, mode="extract"): once you've picked actual dates
from Step 1's printed list, set FLOOD_DATE and BASELINE_DATE below and
re-run in extract mode. This submits the actual export tasks.

DO NOT guess dates and run extract mode directly -- always run
list_dates first and confirm the dates exist.
"""

import argparse
import ee

# ----------------------------------------------------------------
# LOCKED CONSTANTS -- must match training pipeline exactly
# ----------------------------------------------------------------
MAHANADI_AOI = [86.45, 20.15, 86.75, 20.45]
PATCH_GRID_ASSET = "users/mohitpradhan/mahanadi_patch_grid"
GEE_PROJECT = "sar-optical-synthesis"
OUTPUT_CRS = "EPSG:32645"
PATCH_SIZE_M = 2560  # 256px * 10m/px, matches locked decision #3
PATCH_SIZE_PX = 256

# Flood event window, per OSDMA/news reporting: heavy rainfall from
# ~Aug 25 2020, Hirakud Dam gates opened ~Aug 28, inundation continuing
# into early September. Search window deliberately wider than the
# expected peak so list_dates can show what's actually available.
FLOOD_SEARCH_START = "2020-08-20"
FLOOD_SEARCH_END = "2020-09-15"

# Pre-flood baseline candidate window. July 2020, BEFORE the reported
# heavy rainfall onset (~Aug 25), to avoid picking a baseline date that
# is itself already elevated by early monsoon rain. July chosen over
# pre-monsoon (Mar-Apr) deliberately: a Mar-Apr baseline would have
# much LOWER soil moisture / water extent than is normal even for a
# non-flood monsoon month, which would inflate the apparent "new flood
# water" when differenced -- we want a baseline that reflects normal
# WET-SEASON (non-flood) water extent, not dry-season extent. If
# list_dates shows no usable July dates, fall back to early August
# (before Aug 20) rather than jumping all the way back to pre-monsoon.
BASELINE_SEARCH_START = "2020-07-01"
BASELINE_SEARCH_END = "2020-07-31"

# Filled in manually after inspecting list_dates output -- see
# docstring Step 2. Format: "YYYY-MM-DD".
#
# CHOSEN FROM ACTUAL list_dates OUTPUT (only 2 scenes existed in each
# window -- S1's 6-12 day revisit meant there was no flexibility to
# pick an "ideal" date, only a best-available one):
#
# FLOOD_DATE = "2020-08-26"
#   Available flood-window options were 2020-08-26 and 2020-09-07.
#   Per OSDMA/news reporting (Business Standard, ThePrint, India.com):
#   peak Mundali barrage discharge (~10 lakh cusec, the actual
#   medium-to-major flood threshold) hit Sunday Aug 30, with Hirakud
#   gates progressively opened from ~Aug 28; the river was already
#   "swiftly rising" before that. Aug 26 is 4 days PRE-PEAK -- it will
#   show a rising-flood SAR signature, not maximum inundation. This is
#   the honest tradeoff of working with real S1 revisit timing; do NOT
#   describe this date as "peak flood" in the paper, describe it as
#   "rising-stage" or "pre-peak" flood SAR acquisition.
#   Sep 7 was rejected: by then a SEPARATE weather system was already
#   approaching (IMD bulletin referenced a cyclone landfall near Puri
#   on Sep 9), creating risk of a mixed/ambiguous SAR signal that
#   conflates Mahanadi flood recession with a new rain event. No
#   Mahanadi-specific recession data was found confirming Sep 7's
#   water state either way, so Aug 26's unambiguous rising-flood
#   signature was preferred over Sep 7's ambiguous recession-or-new-
#   event signature.
#
# BASELINE_DATE = "2020-07-21"
#   Available baseline-window options were 2020-07-09 and 2020-07-21.
#   2020-07-21 chosen as the LATER (closer-to-flood) date, per the
#   reasoning in BASELINE_SEARCH_START/END comments above: a baseline
#   closer to the flood is more representative of normal pre-flood
#   wet-season water extent than an earlier, possibly-drier date.
FLOOD_DATE = "2020-08-26"
BASELINE_DATE = "2020-07-21"


def init_ee():
    ee.Initialize(project=GEE_PROJECT)


def get_aoi_geometry():
    xmin, ymin, xmax, ymax = MAHANADI_AOI
    return ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])


def get_s1_collection(aoi, start_date, end_date):
    """
    Sentinel-1 GRD, DESCENDING orbit only (locked decision #11),
    IW mode, VV+VH polarization -- same filtering convention as
    02_s1_gap_extraction.py.
    """
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )


def list_available_dates(aoi, start_date, end_date, label):
    """
    Prints every available S1 descending-orbit acquisition date in the
    window. Run this BEFORE extract mode -- do not assume a date
    exists just because it falls in a reasonable-looking window.
    """
    col = get_s1_collection(aoi, start_date, end_date)
    n = col.size().getInfo()
    print(f"\n--- {label}: {n} descending-orbit S1 scene(s) found "
          f"between {start_date} and {end_date} ---")
    if n == 0:
        print("  ZERO scenes found. Widen the search window before proceeding.")
        return []

    dates = col.aggregate_array("system:time_start").getInfo()
    ids = col.aggregate_array("system:index").getInfo()
    out = []
    for d_ms, image_id in zip(dates, ids):
        # convert ms since epoch to YYYY-MM-DD for human inspection
        import datetime
        d_str = datetime.datetime.utcfromtimestamp(d_ms / 1000).strftime("%Y-%m-%d")
        print(f"  {d_str}   (system:index={image_id})")
        out.append(d_str)
    return out


def derive_water_mask(s1_image, vv_threshold_db=-17.0):
    """
    Simple VV-backscatter threshold water mask. Water surfaces are
    smooth and specularly reflect radar away from the sensor, producing
    LOW backscatter; -17 dB is a commonly used starting point in SAR
    flood-mapping literature, but this is a STARTING point, not a
    final value -- it should be sanity-checked against the baseline
    (non-flood) date for this specific AOI before being treated as
    final, since optimal thresholds vary by local surface roughness,
    incidence angle, and sensor calibration. Do not treat -17 dB as
    locked without that check (see module9 validation script, where
    the threshold gets visually/quantitatively sanity-checked against
    known permanent water bodies in the AOI before being applied to
    the flood date).

    Returns a binary mask: 1 = water, 0 = not water.
    """
    vv = s1_image.select("VV")
    water = vv.lt(vv_threshold_db)
    return water.rename("water_mask")


def build_sar_export_image(s1_image):
    """
    Raw VV/VH only, cast to a single explicit shared type (Float32).
    Kept separate from the water mask -- see build_mask_export_image
    docstring for why these cannot be bundled into one multi-band
    image (bug discovered during first extract attempt: GEE refused
    the export with "inconsistent types: Float64 and Byte").
    """
    vv = s1_image.select("VV").toFloat()
    vh = s1_image.select("VH").toFloat()
    return ee.Image.cat([vv, vh]).rename(["VV", "VH"])


def build_mask_export_image(water_mask):
    """
    Derived water mask only, as its own single-band Byte image.

    WHY THIS IS SEPARATE FROM SAR EXPORT (bug fix, found on first real
    extract attempt -- GEE Task ID YNVXY757USKIN7YUR3PQB3Y2 failed with
    "Exported bands must have compatible data types; found inconsistent
    types: Float64 and Byte"):

    VV/VH come back from COPERNICUS/S1_GRD as Float (dB backscatter,
    needs decimal precision, can be negative). water_mask is a binary
    0/1 classification and was cast to Byte for compact storage.
    ee.Image.cat() does NOT auto-promote/unify band types across a
    multi-band image -- every band in one exported image must already
    share an identical type before cat(). Forcing the mask to Float
    just to match VV/VH would work but wastes storage on a binary
    signal and obscures its binary nature downstream; forcing VV/VH to
    Byte would destroy their decimal dB precision entirely. Exporting
    the mask as its own single-band file avoids both bad tradeoffs and
    keeps each file's type self-evidently correct for what it holds.
    """
    return water_mask.toUint8().rename("water_mask")


def export_flood_validation_image(image, date_str, region, description_prefix):

    """
    Export.image.toDrive -- dimensions ONLY, never scale (bug #1).
    CRS EPSG:32645 (bug #2 fix). Matches export conventions used in
    02_s1_gap_extraction.py / 03_s2_optical_extraction.py.
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"{description_prefix}_{date_str}",
        folder="SAR_Optical_FloodValidation",
        region=region,
        dimensions=f"{PATCH_SIZE_PX * 16}x{PATCH_SIZE_PX * 16}",  # placeholder mosaic size,
        # NOTE: this exports a single large mosaic over the whole AOI,
        # NOT per-patch tiles like the training extraction. Re-tiling
        # to the existing mahanadi_patch_grid happens locally in
        # module9 (or via a follow-up per-patch export loop) once you
        # confirm this single-mosaic export looks correct -- exporting
        # 182 individual patch tiles for a one-off validation date is
        # unnecessary engineering for a single validation event; a
        # full-AOI mosaic clipped to the patch grid locally is enough.
        crs=OUTPUT_CRS,
        maxPixels=1e10,
    )
    task.start()
    print(f"Submitted export task: {description_prefix}_{date_str} "
          f"(task id will appear in GEE Tasks tab)")
    return task


def run_list_dates():
    init_ee()
    aoi = get_aoi_geometry()
    print("=" * 70)
    print("STEP 1 -- LISTING AVAILABLE SENTINEL-1 DATES")
    print("Inspect these manually. Do NOT guess dates. Pick real ones,")
    print("then set FLOOD_DATE / BASELINE_DATE constants and re-run with")
    print("--mode extract.")
    print("=" * 70)
    flood_dates = list_available_dates(aoi, FLOOD_SEARCH_START, FLOOD_SEARCH_END,
                                        "FLOOD WINDOW")
    baseline_dates = list_available_dates(aoi, BASELINE_SEARCH_START, BASELINE_SEARCH_END,
                                           "BASELINE WINDOW")

    print("\n" + "=" * 70)
    print("GUIDANCE FOR PICKING DATES:")
    print("- FLOOD_DATE: pick the date closest to peak inundation. Per")
    print("  OSDMA/news reporting, Hirakud Dam gates opened ~Aug 28 2020,")
    print("  with inundation continuing into early September. A date in")
    print("  the Aug 28 - Sep 5 range is most likely to capture peak")
    print("  flood extent, but ONLY if it actually appears in the list")
    print("  above -- do not assume.")
    print("- BASELINE_DATE: pick the LATEST available July date (closer")
    print("  to the flood = more representative of normal pre-flood")
    print("  wet-season water extent). If no July dates are available,")
    print("  widen BASELINE_SEARCH_START backward, or use an early")
    print("  August date strictly before Aug 20.")
    print("=" * 70)

    if not flood_dates or not baseline_dates:
        print("\nWARNING: one or both windows returned zero results.")
        print("Widen FLOOD_SEARCH_START/END or BASELINE_SEARCH_START/END")
        print("constants at the top of this script and re-run.")


def run_extract():
    if FLOOD_DATE is None or BASELINE_DATE is None:
        raise RuntimeError(
            "FLOOD_DATE and BASELINE_DATE are both None. You must run "
            "--mode list_dates FIRST, inspect the printed available "
            "dates, and manually set these two constants at the top of "
            "this script to REAL dates that appeared in that list. "
            "Refusing to guess a date that may not exist (same failure "
            "mode as the original GEE silent-skip bug documented in "
            "module1_build_manifest.py)."
        )

    init_ee()
    aoi = get_aoi_geometry()

    # Each date needs its own +/- 1 day filter window to grab the
    # single matching scene unambiguously.
    def single_date_collection(date_str):
        d = ee.Date(date_str)
        col = get_s1_collection(aoi, d, d.advance(1, "day"))
        n = col.size().getInfo()
        if n == 0:
            raise RuntimeError(
                f"No scene found for {date_str} -- did you copy the "
                f"date exactly as printed by list_dates mode? Aborting "
                f"rather than silently exporting nothing."
            )
        if n > 1:
            print(f"  NOTE: {n} scenes found for {date_str} (overlapping "
                  f"orbit passes); using mosaic of all of them.")
        return col

    print(f"Extracting FLOOD_DATE = {FLOOD_DATE}")
    flood_col = single_date_collection(FLOOD_DATE)
    flood_image = flood_col.mosaic().clip(aoi)
    flood_mask = derive_water_mask(flood_image)
    flood_sar_export = build_sar_export_image(flood_image)
    flood_mask_export = build_mask_export_image(flood_mask)

    print(f"Extracting BASELINE_DATE = {BASELINE_DATE}")
    baseline_col = single_date_collection(BASELINE_DATE)
    baseline_image = baseline_col.mosaic().clip(aoi)
    baseline_mask = derive_water_mask(baseline_image)
    baseline_sar_export = build_sar_export_image(baseline_image)
    baseline_mask_export = build_mask_export_image(baseline_mask)

    # 4 separate export tasks -- SAR (Float32, VV+VH) and mask (Byte,
    # single band) cannot share one file (see build_mask_export_image
    # docstring for the type-mismatch bug this fixes).
    export_flood_validation_image(flood_sar_export, FLOOD_DATE, aoi,
                                   "mahanadi_flood2020_flooddate_sar")
    export_flood_validation_image(flood_mask_export, FLOOD_DATE, aoi,
                                   "mahanadi_flood2020_flooddate_mask")
    export_flood_validation_image(baseline_sar_export, BASELINE_DATE, aoi,
                                   "mahanadi_flood2020_baseline_sar")
    export_flood_validation_image(baseline_mask_export, BASELINE_DATE, aoi,
                                   "mahanadi_flood2020_baseline_mask")

    print("\nAll 4 export tasks submitted (SAR + mask, x2 dates). Check "
          "the GEE Tasks tab (code.earthengine.google.com -> Tasks) for "
          "progress.")
    print("Once all 4 complete and are downloaded (e.g. via rclone, same "
          "as the training data), proceed to module9 for: threshold "
          "sanity-check, baseline differencing, and comparison against "
          "the trained GAN's reconstructed-optical-derived NDWI/MNDWI.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["list_dates", "extract"], required=True,
                         help="list_dates: show available S1 dates, no export. "
                              "extract: submit export tasks for FLOOD_DATE/"
                              "BASELINE_DATE (must be set manually first).")
    args = parser.parse_args()

    if args.mode == "list_dates":
        run_list_dates()
    else:
        run_extract()