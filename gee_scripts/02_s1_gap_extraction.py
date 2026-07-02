"""
GEE SCRIPT 2 (Python) — Sentinel-1 SAR Extraction via earthengine-api
FINAL VERSION — submits all export tasks programmatically, no manual clicking.

Fixes applied:
- Removed `scale` param (was conflicting with `dimensions` — GEE error code 3,
  you cannot specify both; `dimensions` alone fixes output to exactly 256x256 px)
- DESCENDING orbit only (verified: Brahmapur has zero ascending coverage)
- One image per patch per window (earliest descending acquisition)
- CRS = EPSG:32645 (UTM 45N — correct zone for both AOIs, 84°E-90°E)
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

PATCH_SIZE_PX = 256
ORBIT_PASS = 'DESCENDING'
CRS = 'EPSG:32645'

brahmapur_patches = ee.FeatureCollection(f'users/{GEE_USERNAME}/brahmapur_patch_grid')
mahanadi_patches = ee.FeatureCollection(f'users/{GEE_USERNAME}/mahanadi_patch_grid')
all_patches = brahmapur_patches.merge(mahanadi_patches)

def add_window_dates(feature):
    geom = feature.geometry()

    def first_date_for_window(window):
        start, end = window
        coll = (ee.ImageCollection('COPERNICUS/S1_GRD')
                .filterBounds(geom)
                .filterDate(start, end)
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.eq('orbitProperties_pass', ORBIT_PASS))
                .select(['VV', 'VH'])
                .sort('system:time_start'))

        has_images = coll.size().gt(0)
        first_time = ee.Algorithms.If(
            has_images,
            ee.Image(coll.first()).get('system:time_start'),
            -1
        )
        return first_time

    per_window_dates = [first_date_for_window(w) for w in GAP_WINDOWS]
    return feature.set('window_dates', per_window_dates)

patches_with_dates = all_patches.map(add_window_dates)

print('Fetching patch list from server...')
patch_list = patches_with_dates.toList(patches_with_dates.size()).getInfo()
print(f'Patches loaded: {len(patch_list)}')

export_count = 0
skipped_count = 0
failed_submissions = []

for patch_feature in patch_list:
    patch_id = patch_feature['properties']['patch_id']
    window_dates = patch_feature['properties']['window_dates']
    patch_geom = ee.Geometry(patch_feature['geometry'])

    for idx, window in enumerate(GAP_WINDOWS):
        time_ms = window_dates[idx]
        if time_ms == -1 or time_ms is None:
            skipped_count += 1
            continue

        date_str = time.strftime('%Y-%m-%d', time.gmtime(time_ms / 1000))
        start, end = window

        coll = (ee.ImageCollection('COPERNICUS/S1_GRD')
                .filterBounds(patch_geom)
                .filterDate(start, end)
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.eq('orbitProperties_pass', ORBIT_PASS))
                .select(['VV', 'VH'])
                .sort('system:time_start'))

        img = ee.Image(coll.first()).clip(patch_geom)
        description = f's1_{patch_id}_{date_str}'

        try:
            task = ee.batch.Export.image.toDrive(
                image=img,
                description=description,
                folder='sar_optical_dataset/sar_gap_input',
                fileNamePrefix=description,
                region=patch_geom,
                dimensions=f'{PATCH_SIZE_PX}x{PATCH_SIZE_PX}',
                crs=CRS,
                maxPixels=1e9
            )
            task.start()
            export_count += 1
        except Exception as e:
            failed_submissions.append((description, str(e)))

        if export_count % 25 == 0 and export_count > 0:
            print(f'Submitted {export_count} tasks so far...')
            time.sleep(2)

print(f'\nTotal SAR export tasks submitted: {export_count}')
print(f'Patch-windows skipped (no descending SAR coverage): {skipped_count}')
if failed_submissions:
    print(f'Submission-time failures: {len(failed_submissions)}')
    for desc, err in failed_submissions[:5]:
        print(f'  {desc}: {err}')
print('Check progress at https://code.earthengine.google.com/tasks')