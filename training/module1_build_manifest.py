"""
MODULE 1 — Build Triplet Manifest
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
What this does:
  Scans the ACTUAL downloaded files on disk (not the intended GEE export
  list — GEE silently skips exports when no clear scene exists for a
  patch/window, so trusting GEE's own bookkeeping would let missing data
  pass silently). Builds a triplet manifest CSV.

Why triplets, not same-day pairs (locked decision, do not revert):
  A same-day pair (SAR + optical from the SAME date, with synthetic cloud
  pasted on top) lets the model lean on leftover real optical pixels from
  that exact day. At real inference time during an actual monsoon gap,
  there IS NO same-day optical at all — not partial, zero. Training only
  on same-day pairs creates a train/test mismatch. A triplet forces the
  model to bridge a REAL time gap:
    - ref_path:    most recent cloud-free optical BEFORE the gap (input)
    - sar_path:    SAR (VV+VH) acquired DURING the gap window (input)
    - gt_path:     genuinely separate cloud-free optical scene that exists
                   for a date inside/near the gap window (target — held out,
                   never shown to the model as input)

Expected folder structure after downloading from Drive to the NIT PC:
    <DATASET_ROOT>/
        sar_gap_input/        s1_<patch_id>_<date>.tif        (VV,VH — 2 bands)
        optical_reference/    s2ref_<patch_id>_<window_start>.tif  (6 optical + 1 cloud mask = 7 bands)
        optical_ground_truth/ s2gt_<patch_id>_<window_start>.tif   (6 optical bands; MAY be missing)

Output: manifest.csv with columns:
    patch_id, aoi, window_start, sar_date, sar_path, ref_path, gt_path, has_gt

IMPORTANT: only rows with has_gt=True are usable for quantitative
NDWI/NDVI/MNDWI evaluation. Rows with has_gt=False can still be used for
training (the model learns the reconstruction task either way) but must
never enter your reported test-set metrics. This script keeps that
distinction explicit in the manifest itself so it can't be silently mixed
later — exactly the kind of error that caused the Ganjam AOI rework.
"""

import re
import csv
import numpy as np
import rasterio
from pathlib import Path
from datetime import datetime

# ============================================================
# PATHS — set these before running on the NIT PC
# ============================================================
# TODO: point this at wherever you sync/download the Drive folder
DATASET_ROOT = Path("E:/SAR-Optical-Synthesis/data") # <-- SET THIS

SAR_DIR = DATASET_ROOT / "sar_gap_input"
REF_DIR = DATASET_ROOT / "optical_reference"
GT_DIR = DATASET_ROOT / "optical_ground_truth"
OUT_MANIFEST = DATASET_ROOT / "manifest.csv"

# ============================================================
# AOI / GAP WINDOW CONFIG — must match GEE scripts exactly
# ============================================================
# TODO: confirm these match 02_s1_gap_extraction.js / 03_s2_optical_extraction.js
# Both AOIs (Brahmapur/Ganjam corrected coords + Mahanadi delta) feed the
# same manifest structure — the 'aoi' column tags which patch belongs to which.
GAP_WINDOWS = [
    ("2021-06-01", "2021-09-30"),
    ("2022-06-01", "2022-09-30"),
    ("2023-06-01", "2023-09-30"),
]

# Filename patterns
SAR_PATTERN = re.compile(r"s1_(?P<patch_id>.+?)_(?P<date>\d{4}-\d{2}-\d{2})\.tif")
REF_PATTERN = re.compile(r"s2ref_(?P<patch_id>.+?)_(?P<window>\d{4}-\d{2}-\d{2})\.tif")
GT_PATTERN = re.compile(r"s2gt_(?P<patch_id>.+?)_(?P<window>\d{4}-\d{2}-\d{2})\.tif")

# patch_id is expected to encode which AOI it belongs to, e.g.
# "brahmapur_0042" or "mahanadi_0017" — adjust this prefix check if your
# GEE patch-grid script names patches differently.
def infer_aoi(patch_id: str) -> str:
    pid = patch_id.lower()
    if "brahmapur" in pid or "ganjam" in pid:
        return "brahmapur"
    if "mahanadi" in pid:
        return "mahanadi"
    return "unknown"  # TODO: if this shows up in the printed summary, fix patch naming in GEE script 1


# ============================================================
# QUALITY FILTER — restored from the original pipeline's is_valid_patch()
# ============================================================
# The original processing.py rejected any patch with more than 10%
# nodata/NaN/zero pixels BEFORE it ever reached training. That check was
# lost when patch extraction moved from "slice big scenes in Python" to
# "GEE exports pre-sized tiles directly" — this restores the same safety
# net at the point where it now belongs: while building the manifest,
# since this is the first place every downloaded file gets opened anyway.
NODATA_THRESH = 0.10  # reject a file if >10% of its pixels are NaN/zero — same threshold as before

def patch_quality_ok(filepath, nodata_thresh=NODATA_THRESH):
    try:
        with rasterio.open(filepath) as src:
            data = src.read().astype(np.float32)
            n_bands = data.shape[0]
    except Exception as e:
        print(f"WARNING: could not open {filepath} ({e}) — treating as failed quality check")
        return False, 1.0

    # Exclude the last band if this is a reference file (7 bands) — band 6 is
    # CLOUD_MASK which is legitimately all zeros for clear scenes and must not
    # be counted as bad data
    if n_bands == 7:
        data = data[:6]

    total_pixels = data.size
    bad_pixels = np.sum(~np.isfinite(data))  # NaN/Inf only — zeros are valid in SAR and optical
    bad_fraction = bad_pixels / total_pixels
    return bad_fraction < nodata_thresh, float(bad_fraction)

def index_files(directory: Path, pattern: re.Pattern):
    """Returns dict keyed by (patch_id, window_or_date) -> filepath string."""
    index = {}
    if not directory.exists():
        print(f"WARNING: {directory} does not exist yet — did you download/sync this folder?")
        return index
    for f in sorted(directory.glob("*.tif")):
        m = pattern.match(f.name)
        if not m:
            print(f"WARNING: filename did not match expected pattern, skipping: {f.name}")
            continue
        gd = m.groupdict()
        key = (gd["patch_id"], gd.get("window") or gd.get("date"))
        index[key] = str(f)
    return index


def sar_dates_in_window(sar_index, patch_id, window_start, window_end):
    """SAR has multiple acquisitions per window (revisit ~12 days for S1).
    Returns ALL sar file paths for this patch whose date falls inside the
    gap window, sorted chronologically."""
    matches = []
    ws = datetime.strptime(window_start, "%Y-%m-%d")
    we = datetime.strptime(window_end, "%Y-%m-%d")
    for (pid, date_str), path in sar_index.items():
        if pid != patch_id:
            continue
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        if ws <= d <= we:
            matches.append((date_str, path))
    return sorted(matches)


def build_manifest(gap_windows, run_quality_check=True):
    sar_index = index_files(SAR_DIR, SAR_PATTERN)
    ref_index = index_files(REF_DIR, REF_PATTERN)
    gt_index = index_files(GT_DIR, GT_PATTERN)

    print(f"Indexed: {len(sar_index)} SAR files, {len(ref_index)} reference files, "
          f"{len(gt_index)} ground-truth files")

    patch_ids = sorted(set(pid for pid, _ in ref_index.keys()))
    print(f"Found {len(patch_ids)} unique patch_ids with a reference scene")

    if run_quality_check:
        print(f"Running quality check (reject if >{NODATA_THRESH:.0%} NaN/zero pixels)...")
    quality_cache = {}  # filepath -> (ok, bad_fraction), avoids re-reading SAR files reused across windows

    def check_quality(path):
        if not run_quality_check:
            return True
        if path not in quality_cache:
            quality_cache[path] = patch_quality_ok(path)
        return quality_cache[path][0]

    rows = []
    n_with_gt = 0
    n_without_gt = 0
    n_rejected_quality = 0
    aoi_counts = {}

    for patch_id in patch_ids:
        aoi = infer_aoi(patch_id)
        aoi_counts[aoi] = aoi_counts.get(aoi, 0) + 1

        for (start, end) in gap_windows:
            ref_path = ref_index.get((patch_id, start))
            if ref_path is None:
                continue  # no clean pre-gap reference for this patch/window — skip entirely

            if not check_quality(ref_path):
                n_rejected_quality += 1
                continue  # reference file itself is too noisy/incomplete — skip this window for this patch

            gt_path = gt_index.get((patch_id, start), "")
            has_gt = bool(gt_path)
            if has_gt and not check_quality(gt_path):
                # ground truth file exists but fails quality check — treat as if it didn't exist,
                # rather than silently training/evaluating against a corrupted answer key
                n_rejected_quality += 1
                gt_path = ""
                has_gt = False

            if has_gt:
                n_with_gt += 1
            else:
                n_without_gt += 1

            sar_matches = sar_dates_in_window(sar_index, patch_id, start, end)
            if not sar_matches:
                continue  # no SAR acquisition in this window for this patch — skip

            for sar_date, sar_path in sar_matches:
                if not check_quality(sar_path):
                    n_rejected_quality += 1
                    continue  # this specific SAR acquisition is too noisy — skip just this date, keep others

                rows.append({
                    "patch_id": patch_id,
                    "aoi": aoi,
                    "window_start": start,
                    "sar_date": sar_date,
                    "sar_path": sar_path,
                    "ref_path": ref_path,
                    "gt_path": gt_path,
                    "has_gt": has_gt,
                })

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patch_id", "aoi", "window_start", "sar_date",
            "sar_path", "ref_path", "gt_path", "has_gt"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nManifest written: {OUT_MANIFEST}")
    print(f"Total triplet rows: {len(rows)}")
    print(f"  with ground truth (usable for quantitative eval):     {n_with_gt}")
    print(f"  without ground truth (training/qualitative only):    {n_without_gt}")
    if run_quality_check:
        print(f"  files rejected by quality check (>{NODATA_THRESH:.0%} NaN/zero): {n_rejected_quality}")
    print(f"\nPatches per AOI: {aoi_counts}")
    if aoi_counts.get("unknown", 0) > 0:
        print("WARNING: some patch_ids could not be assigned an AOI — fix infer_aoi() "
              "or your GEE patch-naming convention before proceeding.")
    print("\nIMPORTANT: only rows with has_gt=True can go into your reported")
    print("NDWI/NDVI/MNDWI/BSI evaluation table (R2, RMSE, SSIM, shoreline error).")
    print("Module 2 (dataset.py) filters on this automatically via require_gt=True,")
    print("but double-check this count looks reasonable before training — if it's")
    print("near zero, your monsoon-window clear-scene search radius is too tight.")


if __name__ == "__main__":
    build_manifest(GAP_WINDOWS, run_quality_check=True)
