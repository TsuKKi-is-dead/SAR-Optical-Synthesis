# SAR-Optical-Synthesis

# SAR-Guided Optical Reconstruction — Merged Pipeline

This is the single, reconciled pipeline replacing the two earlier,
incompatible drafts (the "synthetic same-day cloud-masking" pipeline and
the "real-triplet gap-filling" pipeline). Read this whole file before
running anything — it records the decisions that were made, WHY they were
made, and exactly what is still a placeholder you must fill in.

## 1. What this pipeline actually does (final, locked framing)

**Task**: SAR + temporally-adjacent cloud-free optical fusion for
monsoon-gap optical reconstruction ("Option B", decided early in this
project). NOT pure SAR-to-optical translation (that would mean SAR-only
input, a harder and differently-benchmarked task — see Section 4).

**Input** (9 channels, 256×256, LOCKED patch size):
| Ch | Content |
|----|---------|
| 0 | VV (SAR, during monsoon gap) |
| 1 | VH (SAR, during monsoon gap) |
| 2-7| B2,B3,B4,B8,B11,B12 — cloud-free optical REFERENCE, most recent clear scene before the gap |
| 8 | Cloud mask of the REFERENCE scene (tells the model where its reference is unreliable, not a synthetic blackout) |

**Target** (6 channels, 256×256): B2,B3,B4,B8,B11,B12 — a REAL, separate,
genuinely cloud-free optical scene from a date inside/near the gap window.
Rare by construction (that scarcity is the reason this project exists).

**Why triplets, not same-day synthetic-cloud pairs** (this was litigated
and decided early in the project, then briefly walked back, then
reinstated — it is correct, do not revert): same-day pairs let the model
lean on leftover real optical pixels from the same date. At true
inference time during a real monsoon gap there is ZERO same-day optical,
not partial — so training only on same-day pairs creates a train/test
mismatch. Triplets force the model to bridge a genuine date gap, which is
the actual deployed problem.

## 2. AOIs — two roles, two regions

- **Brahmapur / Ganjam coast** — general training/test AOI. **MUST use
  the corrected coordinates** (the same fix already validated in the
  submitted erosion paper). This also becomes the AOI for the planned
  follow-up erosion paper (Paper 2), so consistency here matters beyond
  this pipeline.
- **Mahanadi delta** — flood-validation AOI (2020 flood event). Untouched
  by the Ganjam coordinate bug; no change needed.

Both AOIs are processed identically by this pipeline — same bands, same
patch size, same triplet logic. Only the coordinates differ.

## 3. Loss function — why it's uniform, not region-weighted

The composite loss (L1 + perceptual + SSIM) is applied **uniformly**
across the whole output, with NO 0.8/0.2 masked-vs-clear region split.

This is a deliberate correction, not a simplification. The region split
existed in an earlier loss design built for synthetic same-day
cloud-pasting, where part of the image was genuinely already known
(never touched) and part was synthetically hidden — the split protected
the known part and emphasized the hidden part. In the triplet task, the
**entire target is a real scene from a different date the model never
saw** — there is no "easy, already-known" region left to protect. The
0.8/0.2 split has nothing left to attach to, so it was dropped, not
weakened. Full reasoning is in `training/module3_loss.py`'s docstring —
read it before changing this back, and if a reviewer asks, that
docstring is your answer almost verbatim.

## 4. Why a deterministic U-Net, not GAN/diffusion, as primary model

Checked against current literature, not assumed:

- GAN/diffusion methods for SAR-optical synthesis are optimized for
  **perceptual** quality (FID/LPIPS/SSIM-on-image) — visually convincing,
  not guaranteed spectrally correct.
- Documented GAN failure modes for this exact task: training instability,
  mode collapse, color/spectral distortion.
- Diffusion models sample from a learned distribution — the same input
  can yield slightly different output values across runs. For DSAS
  shoreline transects or flood-extent measurement, that sampling noise is
  indistinguishable from real coastal change.
- This pipeline's downstream use is **quantitative** (erosion rate, flood
  extent in physical units), not visualization — so deterministic
  pixel-level regression is the right optimization target, even at the
  cost of potentially blurrier textures than a GAN.
- The GAN is still implemented and trained on **identical** data/splits
  as an honest ablation (`module5_gan_baseline.py`) — this comparison
  itself is part of the paper's contribution, regardless of which wins.

## 5. Novelty positioning — what to tell reviewers

Checked against current literature (as of early 2026):

- **Crowded**: flood mapping directly on SAR (no optical synthesis at
  all) is heavily published — don't position the paper there.
- **Crowded**: SAR-to-optical synthesis via GAN/diffusion for general
  image translation (CloudBreaker, S-CycleGAN, OSCAR, SwinBUFormer, etc.)
  is an active, competitive field — don't claim plain architecture
  novelty here.
- **Open**: SAR+optical fusion specifically validated by **coastal/water
  spectral indices** (NDWI/MNDWI) for monsoon-period gap-filling on an
  Indian coastline, chained into downstream flood/erosion analysis.
  Existing gap-filling-by-index work (CNN-Transformer, Gaussian Process
  fusion) is vegetation-index (NDVI/LAI) over agricultural regions — not
  water indices, not this region.
- CloudBreaker reports NDWI/NDVI SSIM of 0.6156/0.6874 using SAR-only
  input (harder task than yours — note this honestly, don't claim a
  direct win without flagging the easier input). This pipeline reports a
  **four-metric battery** (R², RMSE, SSIM-on-index, shoreline error in
  metres) — broader than any single prior work, and shoreline error in
  metres is the most reviewer-legible number since it's directly
  comparable to the already-submitted erosion paper's convention.

**Your actual contribution sentence**: deterministic SAR+optical fusion
for monsoon-gap reconstruction, evaluated by coastal spectral-index
fidelity (not just image-quality metrics) and chained into a validated
downstream erosion/flood pipeline — an underexplored regional/application
combination, not a new architecture.

## 6. Pipeline structure

```
gee_scripts/
  01_aoi_and_patch_grid.js       AOI polygons + 256x256 patch grid, both AOIs
  02_s1_gap_extraction.js        SAR (VV,VH) during monsoon gap windows
  03_s2_optical_extraction.js    Reference (pre-gap clear) + ground truth (rare in-gap clear)
training/
  module1_build_manifest.py     Scans REAL downloaded files, builds manifest.csv
  module2_dataset.py             PyTorch Dataset — loads triplets, 9ch in / 6ch target
  module3_loss.py                Composite loss (L1+perceptual+SSIM), uniform
  module4_attention_unet.py      Primary model (~31M params, verified)
  module5_gan_baseline.py        GAN ablation (same generator architecture)
  module6_evaluation.py          Full metric battery: R2, RMSE, SSIM-on-index, shoreline error
  module7_train.py               Trains + evaluates both, prints comparison table
```

## 7. What was actually verified vs. not

**Verified by running real code in this session** (not just read/written):

- All seven Python modules parse and import cleanly
- `module3_loss.py`: forward pass + backprop confirmed (VGG perceptual
  weights couldn't download in this sandbox — no internet access to
  pytorch hub from here — but the code path was verified using a
  random-initialized VGG of the same architecture; on the NIT PC with
  internet access, `VGG16_Weights.DEFAULT` will download normally)
- `module4_attention_unet.py`: builds at 256×256, **31,398,834 params
  (~31.4M)** — close to but not exactly your ~35M target; tune `base_ch`
  if the paper needs an exact number, then lock it
- `module5_gan_baseline.py`: one full generator+discriminator training
  step verified on CPU
- `module6_evaluation.py`: sanity-checked against near-identical
  synthetic pred/gt — confirmed RMSE near zero, R² near 1, SSIM near 1,
  as expected
- `module1_build_manifest.py` + `module7_train.py`: **full pipeline
  smoke-tested end-to-end** with synthetic GeoTIFFs matching the exact
  manifest schema — manifest building, AOI tagging, has_gt filtering,
  patch-level train/val/test split, U-Net training (2 epochs), GAN
  training (1 epoch), evaluation, and the final comparison table all ran
  without error on CPU
- `module1_build_manifest.py`'s quality filter (restored from the
  original `processing.py`'s `is_valid_patch()`, which had been dropped
  in the first merge pass — see Section 9 below): verified against a
  deliberately corrupted (all-zero) test file, confirmed it gets rejected
  and excluded from the manifest while a genuine file passes through

**NOT verified (no real data exists yet)**:

- The GEE scripts (`.js` files) cannot be executed outside the GEE code
  editor — they are written to the same conventions as the rest of the
  pipeline but have not been run
- Real-world band order, normalization constants, and actual triplet
  yield (how many patches will really have ground truth) are unknown
  until you run the GEE scripts and look at the printed manifest counts

## 9. What carried over from the original pipeline, and what didn't

When the two earlier pipelines were merged, most preprocessing carried
over correctly (SAR dB normalization, optical /10000 normalization,
NaN/Inf checks, geometric augmentation). One piece was initially dropped
by accident, then restored — recorded here so it doesn't quietly vanish
again in a future edit:

- **Quality/nodata filtering** — the original `processing.py` rejected
  any patch with >10% NaN/zero pixels before it reached training
  (`is_valid_patch()`). This got lost when patch _extraction_ (slicing
  big scenes in Python) was replaced by GEE exporting pre-sized tiles
  directly — there was no longer an obvious place in the code for a
  quality check to live. It's now restored in
  `module1_build_manifest.py` (`patch_quality_ok()`), applied to every
  SAR/reference/ground-truth file individually at manifest-build time,
  since that's the first point every downloaded file gets opened. A
  rejected reference or SAR file drops that triplet row entirely; a
  rejected ground-truth file downgrades the row to `has_gt=False` rather
  than silently evaluating against a corrupted answer key. Verified
  against a deliberately all-zero test file — confirmed rejected.

If you spot another preprocessing step from the old pipeline that seems
to be missing here, it's worth asking explicitly rather than assuming —
this merge process has already missed one real thing once.

## 10. Checklist before running on the NIT PC

1. [ ] Fill in `BRAHMAPUR_AOI_COORDS` in script 1 with the **verified,
       corrected** Ganjam/Brahmapur bounds — copy from the erosion
       paper's locked AOI definition, do not re-derive from memory
2. [ ] Fill in `MAHANADI_AOI_COORDS` in script 1
3. [ ] Fill in `GEE_USERNAME` in all three GEE scripts (must match)
4. [ ] Visually confirm both AOI polygons on the GEE Map panel before
       exporting anything
5. [ ] Confirm `GAP_WINDOWS` is character-for-character identical across
       script 2, script 3, and `module1_build_manifest.py`
6. [ ] Confirm UTM zone (placeholder: EPSG:32644 / UTM 44N) matches your
       actual AOI longitude
7. [ ] After downloading from Drive, confirm folder names exactly match:
       `sar_gap_input/`, `optical_reference/`, `optical_ground_truth/`
8. [ ] Set `DATASET_ROOT` in `module1_build_manifest.py`, run it, and
       **check the printed has_gt count is not near zero** before
       training — if it is, your `MAX_PRE_GAP_SEARCH_DAYS` or
       cloud-probability threshold in script 3 needs loosening
9. [ ] Confirm `OPTICAL_BAND_ORDER` in `module2_dataset.py` and
       `OPTICAL_BANDS` in GEE script 3 are in the exact same order
10. [ ] Confirm whether GEE already divides reflectance by 10000 before
        export, or whether `normalize_optical()` in `module2_dataset.py`
        needs to do it — do not double-apply
11. [ ] Run `python module2_dataset.py /path/to/manifest.csv` standalone
        first and visually check the saved verification figure before
        starting any real training run
12. [ ] If you want the paper's stated parameter count to be exact
        (e.g. ~35M not ~31.4M), tune `base_ch` in
        `module4_attention_unet.py` and re-run its standalone verification
