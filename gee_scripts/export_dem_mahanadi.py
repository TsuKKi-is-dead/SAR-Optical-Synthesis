"""
export_dem_mahanadi.py
========================
One-off: export Copernicus GLO-30 DEM over the Mahanadi AOI, same grid as the
flood SAR/z-score outputs, for visual elevation cross-checking in QGIS.

Usage:
  python gee_scripts\\export_dem_mahanadi.py
"""

import ee

ee.Initialize(project='sar-optical-synthesis')

AOI = ee.Geometry.Rectangle([86.45, 20.15, 86.75, 20.45])
CRS = 'EPSG:32645'
DIMENSIONS = '4096x4096'   # matches flood SAR / z-score mask grid exactly

dem = (
    ee.ImageCollection('COPERNICUS/DEM/GLO30')
    .filterBounds(AOI)
    .select('DEM')
    .mosaic()
    .clip(AOI)
    .toFloat()
)

task = ee.batch.Export.image.toDrive(
    image=dem,
    description='mahanadi_dem_glo30',
    folder='SAR_Optical_FloodValidation',
    fileNamePrefix='mahanadi_dem_glo30',
    dimensions=DIMENSIONS,
    crs=CRS,
    region=AOI,
    maxPixels=1e9,
    fileFormat='GeoTIFF',
)
task.start()
print(f"Export task submitted: mahanadi_dem_glo30")
print(f"Task ID: {task.id}")
print("Monitor at: https://code.earthengine.google.com/tasks")
print("Once complete, download with:")
print(r'  rclone copy gdrive:SAR_Optical_FloodValidation/mahanadi_dem_glo30.tif '
      r'E:\SAR-Optical-Synthesis\data\flood_validation\ --progress')