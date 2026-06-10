# ============================================================
# Module 2 — Preprocessing Pipeline
# SAR-Optical Synthesis Project
# ============================================================
# What this does:
#   1. Loads all _dry_9band.tif files (training pairs)
#   2. Validates band structure (9 bands expected)
#   3. Extracts 256×256 patches with 128px stride (50% overlap)
#   4. Removes patches with >10% nodata
#   5. Applies synthetic cloud augmentation
#   6. Scene-level train/val/test split (70/20/10)
#   7. Saves as .npy files ready for Module 3
#
# Input:  /Users/tsukki/SAR-Optical-Synthesis/data/SAR_Optical_Patches/*_dry_9band.tif
# Output: /Users/tsukki/SAR-Optical-Synthesis/data/processed/
#           train_inputs.npy   (N, 9, 256, 256)
#           train_targets.npy  (N, 6, 256, 256)
#           val_inputs.npy     (N, 9, 256, 256)
#           val_targets.npy    (N, 6, 256, 256)
#           test_inputs.npy    (N, 9, 256, 256)
#           test_targets.npy   (N, 6, 256, 256)
#           split_info.npy     (scene names per split)
# ============================================================

import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────
DATA_DIR   = '/Users/tsukki/SAR-Optical-Synthesis/data/SAR_Optical_Patches'
OUTPUT_DIR = '/Users/tsukki/SAR-Optical-Synthesis/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Parameters ───────────────────────────────────────────────
PATCH_SIZE    = 256
STRIDE        = 128       # 50% overlap
NODATA_THRESH = 0.10      # drop patches with >10% nodata/nan
MAX_PATCHES   = 2500      # cap total patches (subsample if exceeded)
RANDOM_SEED   = 42

# Band layout in 9-band GeoTIFF (0-indexed):
# 0: VV     1: VH
# 2: B2     3: B3     4: B4
# 5: B8     6: B11    7: B12
# 8: cloud_mask
SAR_BANDS     = [0, 1]           # indices
OPTICAL_BANDS = [2, 3, 4, 5, 6, 7]
MASK_BAND     = 8

# ============================================================
# STEP 1 — Find all dry 9-band files
# ============================================================
print('\n=== STEP 1: Finding input files ===')
tif_files = sorted(glob.glob(os.path.join(DATA_DIR, '*_dry_9band.tif')))
print(f'Found {len(tif_files)} dry_9band files:')
for f in tif_files:
    print(f'  {os.path.basename(f)}')

if len(tif_files) == 0:
    raise FileNotFoundError(f'No _dry_9band.tif files found in {DATA_DIR}')

# ============================================================
# STEP 2 — Validate band structure
# ============================================================
print('\n=== STEP 2: Validating band structure ===')
def validate_tif(filepath):
    with rasterio.open(filepath) as src:
        n_bands = src.count
        shape   = (src.height, src.width)
        dtype   = src.dtypes[0]
        nodata  = src.nodata
    return n_bands, shape, dtype, nodata

valid_files = []
for f in tif_files:
    n_bands, shape, dtype, nodata = validate_tif(f)
    name = os.path.basename(f)
    if n_bands != 9:
        print(f'  SKIP {name} — expected 9 bands, got {n_bands}')
        continue
    print(f'  OK   {name} — {n_bands} bands, {shape}, {dtype}')
    valid_files.append(f)

print(f'\n{len(valid_files)} valid files ready for processing')

# ============================================================
# STEP 3 — Load scene and extract patches
# ============================================================
print('\n=== STEP 3: Extracting patches ===')

def load_scene(filepath):
    """Load GeoTIFF and return numpy array (bands, H, W)."""
    with rasterio.open(filepath) as src:
        data = src.read().astype(np.float32)
    return data

def extract_patches(scene, patch_size=256, stride=128):
    """
    Extract patches from scene array (bands, H, W).
    Returns list of patches (bands, patch_size, patch_size).
    """
    _, H, W = scene.shape
    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = scene[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return patches

def is_valid_patch(patch, nodata_thresh=0.10):
    """
    Returns True if patch has less than nodata_thresh fraction
    of NaN or zero pixels across optical bands.
    """
    optical = patch[OPTICAL_BANDS]
    total_pixels = optical.size
    bad_pixels   = np.sum(~np.isfinite(optical)) + np.sum(optical == 0)
    return (bad_pixels / total_pixels) < nodata_thresh

# Extract patches per scene, track which scene each came from
all_patches   = []   # list of (9, 256, 256) arrays
scene_indices = []   # which scene index each patch came from

for scene_idx, filepath in enumerate(tqdm(valid_files, desc='Extracting patches')):
    scene   = load_scene(filepath)
    patches = extract_patches(scene, PATCH_SIZE, STRIDE)
    
    kept = 0
    for patch in patches:
        if is_valid_patch(patch, NODATA_THRESH):
            # Replace any remaining NaN with 0
            patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)
            # Clamp all values to 0-1
            patch = np.clip(patch, 0.0, 1.0)
            all_patches.append(patch)
            scene_indices.append(scene_idx)
            kept += 1
    
    print(f'  {os.path.basename(filepath)}: {len(patches)} total → {kept} kept')

all_patches   = np.array(all_patches, dtype=np.float32)
scene_indices = np.array(scene_indices)
print(f'\nTotal patches before augmentation: {len(all_patches)}')
print(f'Array shape: {all_patches.shape}')

# ============================================================
# STEP 4 — Synthetic cloud augmentation
# ============================================================
print('\n=== STEP 4: Synthetic cloud augmentation ===')

def apply_synthetic_clouds(patch, cloud_masks_pool, rng):
    """
    Take a clear patch and apply a random cloud mask from pool.
    Returns augmented patch with cloud mask channel updated.
    Returns None if donor mask has no clouds (skip augmentation).
    """
    donor_mask = rng.choice(cloud_masks_pool)
    
    # Only augment if donor mask has actual clouds (>5% cloudy)
    cloud_fraction = np.mean(donor_mask > 0.5)
    if cloud_fraction < 0.05:
        return None
    
    aug_patch = patch.copy()
    
    # Apply mask to optical bands — zero out cloudy pixels
    for b in OPTICAL_BANDS:
        aug_patch[b] = np.where(donor_mask > 0.5, 0.0, patch[b])
    
    # Update cloud mask channel with donor mask
    aug_patch[MASK_BAND] = donor_mask
    
    return aug_patch

# Collect all cloud masks from dataset
cloud_masks_pool = all_patches[:, MASK_BAND, :, :]  # (N, 256, 256)

rng = np.random.default_rng(RANDOM_SEED)

# Only augment patches that are currently clear (low cloud fraction)
augmented_patches  = []
augmented_scene_idx = []

for i, patch in enumerate(tqdm(all_patches, desc='Augmenting')):
    current_cloud = np.mean(patch[MASK_BAND] > 0.5)
    
    # Only augment clear patches (cloud fraction < 20%)
    if current_cloud < 0.20:
        aug = apply_synthetic_clouds(patch, cloud_masks_pool, rng)
        if aug is not None:
            augmented_patches.append(aug)
            augmented_scene_idx.append(scene_indices[i])

if len(augmented_patches) > 0:
    augmented_patches   = np.array(augmented_patches, dtype=np.float32)
    augmented_scene_idx = np.array(augmented_scene_idx)
    
    # Combine original + augmented
    all_patches_combined   = np.concatenate([all_patches, augmented_patches], axis=0)
    scene_indices_combined = np.concatenate([scene_indices, augmented_scene_idx], axis=0)
    print(f'Added {len(augmented_patches)} augmented patches')
    print(f'Total after augmentation: {len(all_patches_combined)}')
else:
    all_patches_combined   = all_patches
    scene_indices_combined = scene_indices
    print('No augmentation applied (no clear patches found)')

# ============================================================
# STEP 5 — Subsample if over MAX_PATCHES
# ============================================================
print('\n=== STEP 5: Subsampling ===')

if len(all_patches_combined) > MAX_PATCHES:
    idx = rng.choice(len(all_patches_combined), MAX_PATCHES, replace=False)
    idx = np.sort(idx)
    all_patches_combined   = all_patches_combined[idx]
    scene_indices_combined = scene_indices_combined[idx]
    print(f'Subsampled to {MAX_PATCHES} patches')
else:
    print(f'No subsampling needed — {len(all_patches_combined)} patches total')

# ============================================================
# STEP 6 — Scene-level train/val/test split
# ============================================================
print('\n=== STEP 6: Train/Val/Test split ===')

# Split at scene level to prevent patch leakage
unique_scenes = np.unique(scene_indices_combined)
n_scenes      = len(unique_scenes)

# 70/20/10 split
train_scenes, temp_scenes = train_test_split(
    unique_scenes, test_size=0.30, random_state=RANDOM_SEED)
val_scenes, test_scenes = train_test_split(
    temp_scenes, test_size=0.33, random_state=RANDOM_SEED)

print(f'Total scenes: {n_scenes}')
print(f'Train scenes ({len(train_scenes)}): {[os.path.basename(valid_files[i]) for i in train_scenes]}')
print(f'Val scenes   ({len(val_scenes)}):   {[os.path.basename(valid_files[i]) for i in val_scenes]}')
print(f'Test scenes  ({len(test_scenes)}):  {[os.path.basename(valid_files[i]) for i in test_scenes]}')

def get_split_patches(patches, scene_idx, scene_list):
    mask = np.isin(scene_idx, scene_list)
    return patches[mask]

train_patches = get_split_patches(all_patches_combined, scene_indices_combined, train_scenes)
val_patches   = get_split_patches(all_patches_combined, scene_indices_combined, val_scenes)
test_patches  = get_split_patches(all_patches_combined, scene_indices_combined, test_scenes)

print(f'\nPatch counts:')
print(f'  Train: {len(train_patches)}')
print(f'  Val:   {len(val_patches)}')
print(f'  Test:  {len(test_patches)}')

# ============================================================
# STEP 7 — Separate inputs and targets, save .npy files
# ============================================================
print('\n=== STEP 7: Saving .npy files ===')

# Input:  all 9 channels (SAR + cloudy optical + cloud mask)
# Target: optical bands only (6 channels) — the clean reference
def split_input_target(patches):
    inputs  = patches                          # (N, 9, 256, 256)
    targets = patches[:, OPTICAL_BANDS, :, :]  # (N, 6, 256, 256)
    return inputs, targets

train_inputs, train_targets = split_input_target(train_patches)
val_inputs,   val_targets   = split_input_target(val_patches)
test_inputs,  test_targets  = split_input_target(test_patches)

# Save
np.save(os.path.join(OUTPUT_DIR, 'train_inputs.npy'),  train_inputs)
np.save(os.path.join(OUTPUT_DIR, 'train_targets.npy'), train_targets)
np.save(os.path.join(OUTPUT_DIR, 'val_inputs.npy'),    val_inputs)
np.save(os.path.join(OUTPUT_DIR, 'val_targets.npy'),   val_targets)
np.save(os.path.join(OUTPUT_DIR, 'test_inputs.npy'),   test_inputs)
np.save(os.path.join(OUTPUT_DIR, 'test_targets.npy'),  test_targets)

# Save split info for reference
split_info = {
    'train_scenes': [os.path.basename(valid_files[i]) for i in train_scenes],
    'val_scenes':   [os.path.basename(valid_files[i]) for i in val_scenes],
    'test_scenes':  [os.path.basename(valid_files[i]) for i in test_scenes],
}
np.save(os.path.join(OUTPUT_DIR, 'split_info.npy'), split_info)

print(f'Saved to {OUTPUT_DIR}:')
print(f'  train_inputs.npy  {train_inputs.shape}')
print(f'  train_targets.npy {train_targets.shape}')
print(f'  val_inputs.npy    {val_inputs.shape}')
print(f'  val_targets.npy   {val_targets.shape}')
print(f'  test_inputs.npy   {test_inputs.shape}')
print(f'  test_targets.npy  {test_targets.shape}')

# ============================================================
# STEP 8 — Quick sanity check visualization
# ============================================================
print('\n=== STEP 8: Sanity check visualization ===')

def visualize_sample(inputs, targets, idx=0, save_path=None):
    """Plot SAR | cloudy optical | cloud mask | clean optical."""
    inp = inputs[idx]   # (9, 256, 256)
    tgt = targets[idx]  # (6, 256, 256)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # SAR VV
    axes[0].imshow(inp[0], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('SAR VV (input)')
    axes[0].axis('off')
    
    # Cloudy optical RGB (B4, B3, B2)
    rgb_cloudy = np.stack([inp[4], inp[3], inp[2]], axis=-1)
    rgb_cloudy = np.clip(rgb_cloudy, 0, 1)
    axes[1].imshow(rgb_cloudy)
    axes[1].set_title('Cloudy optical RGB (input)')
    axes[1].axis('off')
    
    # Cloud mask
    axes[2].imshow(inp[8], cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[2].set_title('Cloud mask (red=cloudy)')
    axes[2].axis('off')
    
    # Clean optical RGB target (B4, B3, B2)
    rgb_clean = np.stack([tgt[2], tgt[1], tgt[0]], axis=-1)
    rgb_clean = np.clip(rgb_clean, 0, 1)
    axes[3].imshow(rgb_clean)
    axes[3].set_title('Clean optical RGB (target)')
    axes[3].axis('off')
    
    plt.suptitle(f'Sample patch {idx} — Train set', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved visualization: {save_path}')
    plt.show()

viz_path = os.path.join(OUTPUT_DIR, 'sample_patch_check.png')
visualize_sample(train_inputs, train_targets, idx=0, save_path=viz_path)

# ============================================================
# FINAL SUMMARY
# ============================================================
print('\n=== PREPROCESSING COMPLETE ===')
print(f'Input files processed: {len(valid_files)}')
print(f'Train patches: {len(train_inputs)}')
print(f'Val patches:   {len(val_inputs)}')
print(f'Test patches:  {len(test_inputs)}')
print(f'Total:         {len(train_inputs) + len(val_inputs) + len(test_inputs)}')
print(f'Input shape:   {train_inputs.shape[1:]}  (9 channels, 256×256)')
print(f'Target shape:  {train_targets.shape[1:]} (6 channels, 256×256)')
print(f'\nReady for Module 3 — Dataset and Dataloader')