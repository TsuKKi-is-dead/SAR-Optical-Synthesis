/*
============================================================
GEE SCRIPT 1 — AOI Definition and Patch Grid
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
Defines BOTH AOIs used in this paper:
  - Brahmapur / Ganjam coast: general training/test AOI, and the one that
    feeds the future erosion follow-up paper (Paper 2). MUST use the
    CORRECTED coordinates — the same fix already applied in the
    submitted erosion paper. Do NOT reuse the old inland-shifted bounds.
  - Mahanadi delta: flood-validation AOI (2020 flood event).

Splits each AOI into a grid of 256x256 patches (LOCKED patch size —
matches training/module2_dataset.py exactly) and exports the grid as a
reusable GEE asset, so scripts 2 and 3 reference identical patch
geometries (critical: if the grids don't match exactly between SAR and
optical exports, your triplets will be spatially misaligned).

============================================================
ACTION REQUIRED BEFORE RUNNING
============================================================
1. Replace BRAHMAPUR_AOI_COORDS below with your VERIFIED, CORRECTED
   Ganjam/Brahmapur coastline bounds (the ones already validated in the
   erosion paper — do not re-derive these from memory, copy them from
   that paper's locked AOI definition).
2. Replace MAHANADI_AOI_COORDS with your verified Mahanadi delta bounds
   for the 2020 flood validation case.
3. Replace GEE_USERNAME with your actual GEE username/project path.
4. Visually confirm both polygons on the Map panel before exporting —
   look at actual coastline/delta shape, don't just trust the printed
   coordinates.
*/

// ============================================================
// CONFIG — fill in before running
// ============================================================

// TODO: SET THIS — corrected Ganjam/Brahmapur coastline bounds (same fix as erosion paper)
var BRAHMAPUR_AOI_COORDS = [
  [84.7500, 19.2500],  // SW corner — PLACEHOLDER, replace with verified bounds
  [85.0500, 19.2500],  // SE corner
  [85.0500, 19.5500],  // NE corner
  [84.7500, 19.5500]   // NW corner
];

// TODO: SET THIS — verified Mahanadi delta bounds (2020 flood validation AOI)
var MAHANADI_AOI_COORDS = [
  [86.2000, 20.2000],  // SW corner — PLACEHOLDER, replace with verified bounds
  [86.6000, 20.2000],  // SE corner
  [86.6000, 20.6000],  // NE corner
  [86.2000, 20.6000]   // NW corner
];

var GEE_USERNAME = 'your_gee_username';  // TODO: SET THIS
var PATCH_SIZE_M = 2560;  // 256 pixels * 10m/pixel = 2560m patch footprint — LOCKED to match 256x256

// ============================================================
// Build AOI geometries
// ============================================================

var brahmapurAOI = ee.Geometry.Polygon([BRAHMAPUR_AOI_COORDS]);
var mahanadiAOI = ee.Geometry.Polygon([MAHANADI_AOI_COORDS]);

Map.centerObject(brahmapurAOI, 9);
Map.addLayer(brahmapurAOI, {color: 'red'}, 'Brahmapur AOI (VERIFY THIS VISUALLY)');
Map.addLayer(mahanadiAOI, {color: 'blue'}, 'Mahanadi AOI (VERIFY THIS VISUALLY)');

print('Brahmapur AOI area (km^2):', brahmapurAOI.area().divide(1e6));
print('Mahanadi AOI area (km^2):', mahanadiAOI.area().divide(1e6));
print('STOP: visually confirm both polygons match the actual coastline/delta '
      + 'before proceeding — do not trust coordinates blindly. This exact '
      + 'kind of unverified-AOI mistake caused a full rework on the erosion paper.');

// ============================================================
// Patch grid generator
// ============================================================

function buildPatchGrid(aoi, aoiName) {
  var bounds = aoi.bounds();
  var coords = ee.List(bounds.coordinates().get(0));

  // Project to a metric CRS for accurate metre-based patch sizing
  // TODO: confirm UTM zone — 32644 is UTM 44N, correct for Odisha coast (~85E).
  // If your AOI spans a different UTM zone, update this.
  var UTM_CRS = 'EPSG:32644';

  var projected = aoi.transform(UTM_CRS, 1);
  var projBounds = projected.bounds();
  var projCoords = ee.List(projBounds.coordinates().get(0));

  var xs = projCoords.map(function(c) { return ee.List(c).get(0); });
  var ys = projCoords.map(function(c) { return ee.List(c).get(1); });
  var xMin = ee.Number(xs.reduce(ee.Reducer.min()));
  var xMax = ee.Number(xs.reduce(ee.Reducer.max()));
  var yMin = ee.Number(ys.reduce(ee.Reducer.min()));
  var yMax = ee.Number(ys.reduce(ee.Reducer.max()));

  var nx = xMax.subtract(xMin).divide(PATCH_SIZE_M).ceil();
  var ny = yMax.subtract(yMin).divide(PATCH_SIZE_M).ceil();

  var patches = [];
  var nxVal = nx.getInfo();
  var nyVal = ny.getInfo();
  var xMinVal = xMin.getInfo();
  var yMinVal = yMin.getInfo();

  var patchIdx = 0;
  for (var i = 0; i < nxVal; i++) {
    for (var j = 0; j < nyVal; j++) {
      var px0 = xMinVal + i * PATCH_SIZE_M;
      var py0 = yMinVal + j * PATCH_SIZE_M;
      var patchGeom = ee.Geometry.Rectangle(
        [px0, py0, px0 + PATCH_SIZE_M, py0 + PATCH_SIZE_M],
        UTM_CRS, false
      );
      // Only keep patches that actually intersect the AOI (skip empty corners of the bounding box)
      var intersects = patchGeom.intersects(projected, ee.ErrorMargin(1));
      patches.push(ee.Feature(patchGeom, {
        patch_id: aoiName + '_' + ('0000' + patchIdx).slice(-4),
        aoi: aoiName,
        intersects_aoi: intersects
      }));
      patchIdx++;
    }
  }

  var fc = ee.FeatureCollection(patches);
  // Filter to only patches that intersect the real AOI polygon, not just the bounding box
  fc = fc.filter(ee.Filter.eq('intersects_aoi', true));
  print(aoiName + ' patch grid: ' + fc.size().getInfo() + ' patches (256x256px each)');
  return fc;
}

var brahmapurPatches = buildPatchGrid(brahmapurAOI, 'brahmapur');
var mahanadiPatches = buildPatchGrid(mahanadiAOI, 'mahanadi');

Map.addLayer(brahmapurPatches.style({color: 'orange', fillColor: '00000000'}), {}, 'Brahmapur patch grid');
Map.addLayer(mahanadiPatches.style({color: 'cyan', fillColor: '00000000'}), {}, 'Mahanadi patch grid');

// ============================================================
// Export patch grids as reusable assets — scripts 2 & 3 load these
// ============================================================

Export.table.toAsset({
  collection: brahmapurPatches,
  description: 'brahmapur_patch_grid',
  assetId: 'users/' + GEE_USERNAME + '/brahmapur_patch_grid'
});

Export.table.toAsset({
  collection: mahanadiPatches,
  description: 'mahanadi_patch_grid',
  assetId: 'users/' + GEE_USERNAME + '/mahanadi_patch_grid'
});

print('Run the two Export tasks above from the Tasks tab. '
      + 'Once both assets exist, scripts 2 and 3 will load them by ID.');
