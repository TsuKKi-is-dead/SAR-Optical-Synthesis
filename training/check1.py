"""
Diagnose the checkerboard: check GAN output statistics per tile,
split by quadrant, to see if bottom-right tiles produce different
NDWI/MNDWI distributions than the rest of the AOI.
"""
import os, sys, glob
import numpy as np
import torch
import rasterio
from rasterio.crs import CRS
from rasterio.transform import rowcol

BASE_DIR       = r"E:\SAR-Optical-Synthesis"
FLOOD_SAR_PATH = os.path.join(BASE_DIR, "data", "flood_validation",
                               "mahanadi_flood2020_flooddate_sar_2020-08-26.tif")
OPT_REF_DIR    = os.path.join(BASE_DIR, "data", "optical_reference")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_run2_unweighted")
GAN_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "gan_generator_epoch100.pt")
OPT_REF_WINDOW_START = "2021-06-01"
PATCH_SIZE = 256

sys.path.insert(0, os.path.join(BASE_DIR, "training"))
from module4_attention_unet import AttentionUNet

def normalize_sar(p):
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    return ((np.clip(p, -30.0, 0.0) + 30.0) / 30.0).astype(np.float32)

def normalize_opt(p):
    return np.clip(p.astype(np.float32) / 10000.0, 0.0, 1.0)

# Load SAR
with rasterio.open(FLOOD_SAR_PATH) as src:
    flood_sar = src.read().astype(np.float32)
    profile   = src.profile
    transform_target = src.transform

H, W = profile['height'], profile['width']
crs_target = CRS.from_epsg(32645)

# Build patch index
pattern = os.path.join(OPT_REF_DIR, f"s2ref_mahanadi_*_{OPT_REF_WINDOW_START}.tif")
files = glob.glob(pattern)
index = []
for fpath in files:
    with rasterio.open(fpath) as src:
        if src.crs != crs_target:
            continue
        pt = src.transform
        ph, pw = src.height, src.width
    try:
        r0, c0 = rowcol(transform_target, pt.c, pt.f)
    except:
        continue
    r0, c0 = int(r0), int(c0)
    if r0+ph <= 0 or c0+pw <= 0 or r0 >= H or c0 >= W:
        continue
    index.append((fpath, r0, c0, ph, pw))

def get_optical_tile(tile_r, tile_c):
    buf = np.zeros((7, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    for fpath, r0, c0, ph, pw in index:
        or0 = max(r0, tile_r); or1 = min(r0+ph, tile_r+PATCH_SIZE)
        oc0 = max(c0, tile_c); oc1 = min(c0+pw, tile_c+PATCH_SIZE)
        if or0 >= or1 or oc0 >= oc1:
            continue
        with rasterio.open(fpath) as src:
            region = src.read()[:, or0-r0:or1-r0, oc0-c0:oc1-c0].astype(np.float32)
        buf[:, or0-tile_r:or1-tile_r, oc0-tile_c:oc1-tile_c] = region
    return buf

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentionUNet(in_channels=9, out_channels=6)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.load_state_dict(torch.load(GAN_CHECKPOINT, map_location=device))
model = model.to(device).eval()

print(f"\n{'Tile':>6} {'r':>5} {'c':>5} {'quadrant':>10} {'opt_max':>8} {'out_b3_mean':>12} {'ndwi_mean':>10} {'pct_water':>10}")
print("-"*80)

for r in range(0, H - PATCH_SIZE + 1, PATCH_SIZE):
    for c in range(0, W - PATCH_SIZE + 1, PATCH_SIZE):
        sar_p = flood_sar[:, r:r+PATCH_SIZE, c:c+PATCH_SIZE]
        opt_p = get_optical_tile(r, c)

        quadrant = ("BR" if r >= H//2 and c >= W//2 else
                    "BL" if r >= H//2 else
                    "TR" if c >= W//2 else "TL")

        sar_n = normalize_sar(sar_p)
        opt_n = normalize_opt(opt_p)
        inp   = np.concatenate([sar_n, opt_n], axis=0)

        with torch.no_grad():
            out = model(torch.from_numpy(inp).unsqueeze(0).float().to(device))
        out_np = out.squeeze(0).cpu().numpy()

        b3 = out_np[1]; b8 = out_np[3]
        denom = b3 + b8; denom = np.where(denom == 0, 1e-6, denom)
        ndwi = (b3 - b8) / denom
        pct_water = (ndwi > 0).mean() * 100

        print(f"{r//PATCH_SIZE*16+c//PATCH_SIZE:>6} {r:>5} {c:>5} {quadrant:>10} "
              f"{opt_p.max():>8.1f} {out_np[1].mean():>12.4f} "
              f"{ndwi.mean():>10.4f} {pct_water:>9.1f}%")