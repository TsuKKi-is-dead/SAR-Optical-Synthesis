/*
============================================================
GEE SCRIPT 3 — Sentinel-2 Optical Extraction (Reference + Ground Truth)
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
For every patch in both grids, pulls TWO separate optical scenes:

  1. REFERENCE (input) — the most recent cloud-free Sentinel-2 scene
     BEFORE each gap window starts. 6 optical bands + 1 cloud-mask band
     (7 bands total) — the cloud mask here describes clouds in the
     REFERENCE scene itself (used as an input feature in Python, telling
     the model where its reference is unreliable), NOT a synthetic
     blackout.

  2. GROUND TRUTH (target) — a genuinely separate cloud-free Sentinel-2
     scene that exists for a date INSIDE or very near the gap window.
     These are RARE by definition (that's the whole reason this project
     exists) — most patches will have NO ground truth for most windows.
     That's expected and fine; module1_build_manifest.py explicitly
     tracks has_gt=True/False and only has_gt=True rows enter your
     reported quantitative metrics.

Filenames:
  s2ref_<patch_id>_<window_start>.tif   (7 bands)
  s2gt_<patch_id>_<window_start>.tif    (6 bands, only written if found)

These MUST land in folders named exactly optical_reference/ and
optical_ground_truth/ after downloading, matching
module1_build_manifest.py's expected structure.

ACTION REQUIRED: GAP_WINDOWS must exactly match script 2 and
module1_build_manifest.py.
*/

var GEE_USERNAME = 'your_gee_username';  // TODO: SET THIS — must match scripts 1 & 2

var GAP_WINDOWS = [
  ['2021-06-01', '2021-09-30'],
  ['2022-06-01', '2022-09-30'],
  ['2023-06-01', '2023-09-30']
];

var PATCH_SIZE_PX = 256;  // LOCKED
var SCALE_M = 10;

var OPTICAL_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];
// TODO: confirm this exact order matches training/module2_dataset.py's
// OPTICAL_BAND_ORDER — a mismatch here silently corrupts every NDWI/NDVI
// value computed downstream.

var CLOUD_PROB_THRESHOLD = 30;  // s2cloudless probability % above which a pixel is "cloudy"
var MAX_PRE_GAP_SEARCH_DAYS = 90;  // how far back to search for a clear reference scene

var brahmapurPatches = ee.FeatureCollection('users/' + GEE_USERNAME + '/brahmapur_patch_grid');
var mahanadiPatches = ee.FeatureCollection('users/' + GEE_USERNAME + '/mahanadi_patch_grid');
var allPatches = brahmapurPatches.merge(mahanadiPatches);

// ============================================================
// Cloud masking using s2cloudless
// ============================================================

function getS2WithCloudProb(aoi, startDate, endDate) {
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(startDate, endDate)
    .select(OPTICAL_BANDS);

  var s2Cloud = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    .filterBounds(aoi)
    .filterDate(startDate, endDate);

  var joined = ee.Join.saveFirst('cloud_prob').apply({
    primary: s2,
    secondary: s2Cloud,
    condition: ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'})
  });

  return ee.ImageCollection(joined).map(function(img) {
    var cloudProb = ee.Image(img.get('cloud_prob')).select('probability');
    var cloudMask = cloudProb.gte(CLOUD_PROB_THRESHOLD).rename('CLOUD_MASK');
    var cloudFraction = cloudMask.reduceRegion({
      reducer: ee.Reducer.mean(), geometry: aoi, scale: SCALE_M, maxPixels: 1e8
    }).get('CLOUD_MASK');
    return img.addBands(cloudMask).set('cloud_fraction', cloudFraction);
  });
}

function findClearestScene(collection) {
  // Sort by cloud_fraction ascending, take the clearest
  return collection.sort('cloud_fraction').first();
}

// ============================================================
// Per-patch, per-window export loop
// ============================================================

var patchList = allPatches.toList(allPatches.size()).getInfo();
print('Patches to process:', patchList.length);

var refExportCount = 0;
var gtExportCount = 0;
var gtFoundCount = 0;
var gtMissingCount = 0;

patchList.forEach(function(patchFeature) {
  var patchId = patchFeature.properties.patch_id;
  var patchGeom = ee.Geometry(patchFeature.geometry);

  GAP_WINDOWS.forEach(function(window) {
    var windowStart = window[0];
    var windowEnd = window[1];

    // --- 1. Reference scene: clearest in the MAX_PRE_GAP_SEARCH_DAYS before windowStart ---
    var refSearchStart = ee.Date(windowStart).advance(-MAX_PRE_GAP_SEARCH_DAYS, 'day');
    var refCandidates = getS2WithCloudProb(patchGeom, refSearchStart, windowStart);
    var refCount = refCandidates.size().getInfo();

    if (refCount === 0) {
      print('WARNING: no reference scene found for ' + patchId + ' / ' + windowStart + ' — skipping window entirely');
      return;  // no reference -> no usable triplet for this patch/window, matches module1's skip logic
    }

    var refScene = findClearestScene(refCandidates);
    var refImg = ee.Image(refScene).select(OPTICAL_BANDS.concat(['CLOUD_MASK']));

    Export.image.toDrive({
      image: refImg.clip(patchGeom),
      description: 's2ref_' + patchId + '_' + windowStart,
      folder: 'sar_optical_dataset/optical_reference',
      fileNamePrefix: 's2ref_' + patchId + '_' + windowStart,
      region: patchGeom,
      scale: SCALE_M,
      dimensions: PATCH_SIZE_PX + 'x' + PATCH_SIZE_PX,
      crs: 'EPSG:32644',
      maxPixels: 1e9
    });
    refExportCount++;

    // --- 2. Ground truth: rare clear scene INSIDE the gap window itself ---
    var gtCandidates = getS2WithCloudProb(patchGeom, windowStart, windowEnd)
      .filter(ee.Filter.lt('cloud_fraction', 0.05));  // require genuinely clear, <5% cloud
    var gtCount = gtCandidates.size().getInfo();

    if (gtCount === 0) {
      gtMissingCount++;
      return;  // expected and common — this window/patch has no real GT, has_gt=False in manifest
    }

    var gtScene = findClearestScene(gtCandidates);
    var gtImg = ee.Image(gtScene).select(OPTICAL_BANDS);

    Export.image.toDrive({
      image: gtImg.clip(patchGeom),
      description: 's2gt_' + patchId + '_' + windowStart,
      folder: 'sar_optical_dataset/optical_ground_truth',
      fileNamePrefix: 's2gt_' + patchId + '_' + windowStart,
      region: patchGeom,
      scale: SCALE_M,
      dimensions: PATCH_SIZE_PX + 'x' + PATCH_SIZE_PX,
      crs: 'EPSG:32644',
      maxPixels: 1e9
    });
    gtExportCount++;
    gtFoundCount++;
  });
});

print('Reference scene exports queued:', refExportCount);
print('Ground-truth scene exports queued:', gtExportCount);
print('Windows with NO ground truth found (expected, tracked as has_gt=False):', gtMissingCount);
print('Run all tasks from the Tasks tab. After downloading, place files in '
      + 'folders named exactly: optical_reference/ and optical_ground_truth/');
