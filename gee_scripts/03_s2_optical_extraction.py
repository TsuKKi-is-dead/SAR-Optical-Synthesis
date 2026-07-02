"""
GEE SCRIPT 3 (Python) — Sentinel-2 Optical Reference + Ground Truth Extraction
via earthengine-api

Exports two sets of files per patch per gap window:
  1. REFERENCE: clearest S2 scene in the 90 days BEFORE window start
       7 bands: B2, B3, B4, B8, B11, B12, CLOUD_MASK
       filename: s2ref_<patch_id>_<window_start>.tif
       folder:   sar_optical_dataset/optical_reference/

  2. GROUND TRUTH: clearest S2 scene STRICTLY INSIDE the gap window
       with cloud_fraction < 0.05 (sparse by design — most will be skipped)
       6 bands: B2, B3, B4, B8, B11, B12
       filename: s2gt_<patch_id>_<window_start>.tif
       folder:   sar_optical_dataset/optical_ground_truth/

Band order matches module2_dataset.py exactly:
  OPTICAL_BAND_ORDER = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
  Reference export: optical bands first (indices 0-5), CLOUD_MASK last (index 6)
  Ground truth export: optical bands only (indices 0-5)

Fixes carried over from Script 2:
  - dimensions only, no scale (avoids GEE Error code 3)
  - CRS = EPSG:32645 (UTM 45N, correct for both AOIs)
  - Chunked .getInfo() calls (CHUNK_SIZE=15) to avoid timeout on
    expensive per-patch cloud-probability reduceRegion calls
"""

import ee
import time

ee.Initialize(project='sar-optical-synthesis')

GEE_USERNAME = 'mohitpradhan'

GAP_WINDOWS = [
    ('2021-06-01', '2021-09-30'),
    ('2022-06-01', '2022-09-30'),
    ('2023-06-01', '2023-09-30'),
]

OPTICAL_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']  # matches module2_dataset.py OPTICAL_BAND_ORDER exactly
PATCH_SIZE_PX = 256
CRS = 'EPSG:32645'
MAX_PRE_GAP_SEARCH_DAYS = 90
GT_MAX_CLOUD_FRACTION = 0.05
CHUNK_SIZE = 15

brahmapur_patches = ee.FeatureCollection(f'users/{GEE_USERNAME}/brahmapur_patch_grid')
mahanadi_patches = ee.FeatureCollection(f'users/{GEE_USERNAME}/mahanadi_patch_grid')
all_patches = brahmapur_patches.merge(mahanadi_patches)

# ── helpers ────────────────────────────────────────────────────────────────────

def get_s2_with_cloud_prob(start, end, geom):
    s2_sr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(geom)
             .filterDate(start, end)
             .select(OPTICAL_BANDS))

    s2_cloud = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                .filterBounds(geom)
                .filterDate(start, end))

    joined = ee.Join.saveFirst('cloud_prob_img').apply(
        primary=s2_sr,
        secondary=s2_cloud,
        condition=ee.Filter.equals(
            leftField='system:index',
            rightField='system:index'
        )
    )

    def add_cloud_band(obj):
        img = ee.Image(obj)  # explicit cast — Join.saveFirst returns generic objects
        cloud_img = ee.Image(img.get('cloud_prob_img')).select('probability')
        cloud_mask = cloud_img.gt(40).rename('CLOUD_MASK').toUint16()
        return img.addBands(cloud_mask)

    return ee.ImageCollection(joined.map(add_cloud_band))

    def add_cloud_band(img):
        cloud_img = ee.Image(img.get('cloud_prob_img')).select('probability')
        cloud_mask = cloud_img.gt(40).rename('CLOUD_MASK')  # threshold: >40% prob = cloudy pixel
        return img.addBands(cloud_mask)

    return ee.ImageCollection(joined.map(add_cloud_band))


def compute_patch_cloud_fraction(img, geom):
    """
    Computes mean cloud probability over the patch geometry.
    Returns a server-side number (not yet evaluated).
    """
    cloud_prob_img = ee.Image(img.get('cloud_prob_img')).select('probability')
    mean_dict = cloud_prob_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=10,
        maxPixels=1e6
    )
    return mean_dict.getNumber('probability')


def get_best_reference(geom, window_start):
    """
    Finds the clearest (lowest mean cloud probability over patch) S2 scene
    in the MAX_PRE_GAP_SEARCH_DAYS before window_start.
    Returns dict with keys: found (bool), date (str), cloud_fraction (float)
    — evaluated server-side via getInfo().
    """
    end_date = window_start
    start_date = (ee.Date(window_start)
                  .advance(-MAX_PRE_GAP_SEARCH_DAYS, 'day')
                  .format('YYYY-MM-dd')
                  .getInfo())

    coll = get_s2_with_cloud_prob(start_date, end_date, geom)
    size = coll.size().getInfo()
    if size == 0:
        return {'found': False, 'date': None, 'cloud_fraction': None}

    # Sort by scene-level CLOUDY_PIXEL_PERCENTAGE first (cheap metadata sort)
    # then refine with per-patch reduceRegion on the top candidate only
    # This avoids running reduceRegion on every scene in the collection
    sorted_coll = coll.sort('CLOUDY_PIXEL_PERCENTAGE')
    best_img = ee.Image(sorted_coll.first())

    cf = compute_patch_cloud_fraction(best_img, geom).getInfo()
    if cf is None:
        return {'found': False, 'date': None, 'cloud_fraction': None}

    date_ms = best_img.get('system:time_start').getInfo()
    date_str = time.strftime('%Y-%m-%d', time.gmtime(date_ms / 1000))
    return {'found': True, 'date': date_str, 'cloud_fraction': cf}


def get_best_ground_truth(geom, window_start, window_end):
    """
    Finds the clearest S2 scene strictly inside the gap window
    with mean patch cloud fraction < GT_MAX_CLOUD_FRACTION (0.05).
    Returns dict with keys: found (bool), date (str), cloud_fraction (float).
    """
    coll = get_s2_with_cloud_prob(window_start, window_end, geom)
    size = coll.size().getInfo()
    if size == 0:
        return {'found': False, 'date': None, 'cloud_fraction': None}

    sorted_coll = coll.sort('CLOUDY_PIXEL_PERCENTAGE')

    # Iterate through candidates until one passes the per-patch threshold
    # Limit to top 5 candidates to avoid excessive getInfo() calls
    candidates = sorted_coll.toList(5).getInfo()
    for candidate in candidates:
        img = ee.Image(sorted_coll.toList(5).get(candidates.index(candidate)))
        cf = compute_patch_cloud_fraction(img, geom).getInfo()
        if cf is not None and cf < (GT_MAX_CLOUD_FRACTION * 100):  # cloud_prob is 0-100
            date_ms = candidate['properties']['system:time_start']
            date_str = time.strftime('%Y-%m-%d', time.gmtime(date_ms / 1000))
            return {'found': True, 'date': date_str, 'cloud_fraction': cf}

    return {'found': False, 'date': None, 'cloud_fraction': None}


# ── main loop ──────────────────────────────────────────────────────────────────

print('Fetching patch list from server...')
patch_list_raw = all_patches.toList(all_patches.size()).getInfo()
print(f'Patches loaded: {len(patch_list_raw)}')

ref_export_count = 0
gt_export_count = 0
ref_skipped_count = 0
gt_skipped_count = 0
failed_submissions = []

# Chunk the patch list to avoid timeout on expensive per-patch cloud queries
chunks = [patch_list_raw[i:i+CHUNK_SIZE] for i in range(0, len(patch_list_raw), CHUNK_SIZE)]
print(f'Processing {len(chunks)} chunks of up to {CHUNK_SIZE} patches each...')

for chunk_idx, chunk in enumerate(chunks):
    print(f'\nChunk {chunk_idx+1}/{len(chunks)} ({len(chunk)} patches)...')

    for patch_feature in chunk:
        patch_id = patch_feature['properties']['patch_id']
        patch_geom = ee.Geometry(patch_feature['geometry'])

        for window_start, window_end in GAP_WINDOWS:

            # ── REFERENCE export ───────────────────────────────────────────
            ref_info = get_best_reference(patch_geom, window_start)

            if not ref_info['found']:
                ref_skipped_count += 1
                print(f'  REF SKIP: {patch_id} {window_start} — no usable scene in prior 90 days')
            else:
                ref_date = ref_info['date']
                ref_coll = get_s2_with_cloud_prob(
                    (ee.Date(window_start)
                     .advance(-MAX_PRE_GAP_SEARCH_DAYS, 'day')
                     .format('YYYY-MM-dd')
                     .getInfo()),
                    window_start,
                    patch_geom
                )
                ref_img = ee.Image(ref_coll.sort('CLOUDY_PIXEL_PERCENTAGE').first())

                # Export: 6 optical bands + CLOUD_MASK (7 bands total, in order)
                # module2_dataset.py reads ref[:6] as optical, ref[6:7] as cloud mask
                ref_export_img = ref_img.select(OPTICAL_BANDS + ['CLOUD_MASK']).clip(patch_geom)
                description = f's2ref_{patch_id}_{window_start}'

                try:
                    task = ee.batch.Export.image.toDrive(
                        image=ref_export_img,
                        description=description,
                        folder='sar_optical_dataset/optical_reference',
                        fileNamePrefix=description,
                        region=patch_geom,
                        dimensions=f'{PATCH_SIZE_PX}x{PATCH_SIZE_PX}',
                        crs=CRS,
                        maxPixels=1e9
                    )
                    task.start()
                    ref_export_count += 1
                except Exception as e:
                    failed_submissions.append((description, str(e)))

            # ── GROUND TRUTH export ────────────────────────────────────────
            gt_info = get_best_ground_truth(patch_geom, window_start, window_end)

            if not gt_info['found']:
                gt_skipped_count += 1
                # Expected — most patch/windows will have no clear scene during monsoon
            else:
                gt_date = gt_info['date']
                gt_coll = get_s2_with_cloud_prob(window_start, window_end, patch_geom)
                gt_img = ee.Image(gt_coll.sort('CLOUDY_PIXEL_PERCENTAGE').first())

                gt_export_img = gt_img.select(OPTICAL_BANDS).clip(patch_geom)
                description = f's2gt_{patch_id}_{window_start}'

                try:
                    task = ee.batch.Export.image.toDrive(
                        image=gt_export_img,
                        description=description,
                        folder='sar_optical_dataset/optical_ground_truth',
                        fileNamePrefix=description,
                        region=patch_geom,
                        dimensions=f'{PATCH_SIZE_PX}x{PATCH_SIZE_PX}',
                        crs=CRS,
                        maxPixels=1e9
                    )
                    task.start()
                    gt_export_count += 1
                except Exception as e:
                    failed_submissions.append((description, str(e)))

        # Small sleep every patch to avoid hammering the API
        time.sleep(0.5)

    print(f'  Chunk {chunk_idx+1} done. Ref exports so far: {ref_export_count}, GT exports: {gt_export_count}')
    time.sleep(2)  # brief pause between chunks

# ── summary ───────────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'REFERENCE exports submitted:  {ref_export_count}')
print(f'REFERENCE skipped (no scene): {ref_skipped_count}')
print(f'GROUND TRUTH exports submitted: {gt_export_count}')
print(f'GROUND TRUTH skipped (no clear scene in window): {gt_skipped_count}')
if failed_submissions:
    print(f'\nSubmission-time failures: {len(failed_submissions)}')
    for desc, err in failed_submissions[:10]:
        print(f'  {desc}: {err}')
print('\nCheck progress at https://code.earthengine.google.com/tasks')