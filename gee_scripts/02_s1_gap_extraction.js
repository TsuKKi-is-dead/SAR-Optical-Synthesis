/*
============================================================
GEE SCRIPT 2 — Sentinel-1 SAR Extraction (Gap-Window Acquisitions)
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
Pulls VV+VH SAR acquisitions DURING each monsoon gap window, for every
patch in BOTH patch grids (Brahmapur + Mahanadi) exported by script 1.
This is the SAR input half of each triplet (paired later in Python by
module1_build_manifest.py against the optical reference + ground truth).

Filenames written: s1_<patch_id>_<acquisition_date>.tif (2 bands: VV, VH)
These MUST land in a folder named exactly sar_gap_input/ after you
download from Drive, to match module1_build_manifest.py's expected
structure.

ACTION REQUIRED: GAP_WINDOWS below MUST be character-for-character
identical to the windows used in script 3 and in
training/module1_build_manifest.py's GAP_WINDOWS constant. A mismatch
here is a silent, hard-to-debug bug — triplets would simply fail to
match in the manifest builder with no error, just an unexpectedly low
row count.
*/

var GEE_USERNAME = 'your_gee_username';  // TODO: SET THIS — must match script 1

// TODO: keep identical to script 3 and module1_build_manifest.py
var GAP_WINDOWS = [
  ['2021-06-01', '2021-09-30'],
  ['2022-06-01', '2022-09-30'],
  ['2023-06-01', '2023-09-30']
];

var PATCH_SIZE_PX = 256;  // LOCKED
var SCALE_M = 10;         // Sentinel-1 GRD native resolution

// Load the patch grids exported by script 1
var brahmapurPatches = ee.FeatureCollection('users/' + GEE_USERNAME + '/brahmapur_patch_grid');
var mahanadiPatches = ee.FeatureCollection('users/' + GEE_USERNAME + '/mahanadi_patch_grid');
var allPatches = brahmapurPatches.merge(mahanadiPatches);

print('Total patches to process (both AOIs):', allPatches.size());

// ============================================================
// SAR collection — VV+VH, IW mode, both orbit passes
// ============================================================

function getS1ForWindow(aoi, startDate, endDate) {
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .select(['VV', 'VH']);
}

// ============================================================
// Export loop — one image per (patch, acquisition date) pair
// ============================================================
// NOTE: GEE client-side loops over .getInfo() lists are fine for a few
// hundred patches but will be slow for thousands — if you have a very
// large patch grid, batch this script by AOI or by gap-window year to
// avoid browser/console timeouts (this is the reason the original
// pipeline was already split into focused modules).

var patchList = allPatches.toList(allPatches.size()).getInfo();
print('Patch list loaded client-side:', patchList.length);

var exportCount = 0;

patchList.forEach(function(patchFeature) {
  var patchId = patchFeature.properties.patch_id;
  var patchGeom = ee.Geometry(patchFeature.geometry);

  GAP_WINDOWS.forEach(function(window) {
    var startDate = window[0];
    var endDate = window[1];

    var s1Collection = getS1ForWindow(patchGeom, startDate, endDate);
    var dateList = s1Collection.aggregate_array('system:time_start');

    // Client-side date list per patch/window — keeps things simple and
    // matches module1_build_manifest.py's "scan what's really there"
    // philosophy rather than trusting an intended export list.
    var nImages = s1Collection.size().getInfo();
    if (nImages === 0) {
      return;  // no SAR acquisition in this window for this patch — skip, do not export empty
    }

    var imageList = s1Collection.toList(nImages);
    for (var i = 0; i < nImages; i++) {
      var img = ee.Image(imageList.get(i));
      var dateStr = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo();

      Export.image.toDrive({
        image: img.clip(patchGeom),
        description: 's1_' + patchId + '_' + dateStr,
        folder: 'sar_optical_dataset/sar_gap_input',
        fileNamePrefix: 's1_' + patchId + '_' + dateStr,
        region: patchGeom,
        scale: SCALE_M,
        dimensions: PATCH_SIZE_PX + 'x' + PATCH_SIZE_PX,
        crs: 'EPSG:32644',
        maxPixels: 1e9
      });
      exportCount++;
    }
  });
});

print('Total SAR export tasks queued:', exportCount);
print('Run all tasks from the Tasks tab. After downloading from Drive, '
      + 'place files in a folder named exactly: sar_gap_input/');
