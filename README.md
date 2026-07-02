# 🌊 Monsoon Coastal Monitoring via SAR-to-Optical cGAN

<div align="center">

**Reconstructing Monsoon-Season Coastal Land Cover and Shoreline Position from Sentinel-1 SAR Using a Conditional Generative Adversarial Network**

*A Case Study of the Brahmapur–Ganjam Coastline, Odisha, India*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-Sentinel--1%2F2-4285F4?logo=google)](https://earthengine.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange)]()

</div>

---

> **The Problem:** The southwest monsoon (June–September) is the period of *peak coastal hazard* on India's Bay of Bengal coast — yet optical satellite monitoring is effectively impossible during these months owing to near-complete cloud cover. No monsoon-season shoreline or land cover map has ever existed for the Ganjam coastline.
>
> **This study fills that gap.**

---

## 📌 Overview

This repository contains the full pipeline for a **conditional GAN (cGAN)** framework trained to synthesise cloud-free Sentinel-2 multispectral imagery from contemporaneous **Sentinel-1 SAR inputs**, enabling the first ever monsoon-onset shoreline and land cover record for the **Brahmapur–Ganjam coastline, Odisha, India**.

The model uses an **Attention U-Net** generator paired with a **70×70 PatchGAN** discriminator. Applied to three monsoon-onset dates (2021-06-03, 2022-06-10, 2023-06-05), reconstructed imagery was used to extract shorelines and classify coastal land cover — independently validated against real Sentinel-2 ground truth.

This is a **companion study** to:
> Pradhan M. (2025a). *Multi-temporal shoreline change analysis and land cover dynamics of the Ganjam coastline, Odisha, India (2013–2024): a remote sensing and DSAS approach.* [submitted]

---

## 🏆 Key Results

| Metric | U-Net (Ablation) | **cGAN (Primary)** |
|--------|-----------------|-------------------|
| PSNR (dB) | 29.17 | **33.30** |
| SSIM | 0.818 | **0.873** |
| NDWI R² | 0.075 | **0.776** |
| MNDWI R² | 0.032 | **0.797** |
| NDVI R² | −0.328 | **0.734** |

> The U-Net baseline NDWI R² ≈ 0.075 would produce physically meaningless shoreline extractions. **Adversarial training is not a perceptual refinement — it is structurally necessary.**

**Monsoon Shoreline Signal (119 DSAS transects):**

| Year | n valid | Mean displacement (m) | Std (m) |
|------|---------|-----------------------|---------|
| 2021 | 78 | −47.2 | 220.4 |
| 2022 | 80 | −95.5 | 187.2 |
| 2023 | 72 | −43.4 | 213.0 |
| Post-monsoon baseline | 309 | 37.2% landward | — |

**80.0% of monsoon-onset shorelines were landward of post-monsoon counterparts** (post-monsoon baseline: 37.2%). Validated against ground truth: **80.0% (GAN) vs. 79.0% (GT)** — a one-percentage-point difference across 100 matched transect-year observations.

**Monsoon Land Cover (MNDWI-threshold, km²):**

| Year | Land | Intertidal | Water | Total |
|------|------|------------|-------|-------|
| 2021 | 49.516 | 7.135 | 8.582 | 65.233 |
| 2022 | 48.405 | 7.141 | 9.688 | 65.234 |
| **2023** | **46.044** | **13.342** | **5.848** | **65.234** |

---

## 🗺️ Study Area

<!-- Replace with your AOI map image -->
![Study Area](assets/fig_study_area.png)
*Brahmapur–Ganjam coastline, Odisha, India (19.22°N–19.39°N, 84.84°E–85.06°E). ~60 km segment of the Bay of Bengal shore including the Rushikulya River mouth, a critical Olive Ridley sea turtle mass-nesting site.*

---

## 🏗️ Architecture

<!-- Replace with your architecture diagram (Fig. 1 from paper) -->
![Model Architecture](assets/fig1_architecture.png)
*Conditional GAN architecture. **Generator**: Attention U-Net (encoder widths 64/128/256/512, 1024-ch bottleneck, attention-gated skip connections). **Discriminator**: 70×70 PatchGAN classifying overlapping patches as real or generated.*

### Why This Architecture?

- **Why U-Net over a plain encoder–decoder?** Skip connections carry high-resolution spatial structure directly to the decoder, preserving sharp water–land boundaries that shoreline extraction requires. A bottleneck-only design loses this detail.
- **Why Attention Gates over standard skips?** Standard U-Net passes all encoder features indiscriminately, including irrelevant inland vegetation. Attention gates (Oktay et al. 2018) suppress these, concentrating decoder capacity on the beach face and intertidal zone — precisely the pixels that determine shoreline position.
- **Why PatchGAN over a full-image discriminator?** A full-image discriminator assesses realism at scene level, too coarse for local spectral contrast at the water–land boundary. The 70×70 PatchGAN penalises any locally unrealistic patch, which is directly responsible for recovering water-index fidelity (NDWI R² 0.776 vs. 0.075).
- **Why not Transformers or Diffusion models?** Both require far more data than the 311 within-domain paired patches available here. Transformer self-attention scales quadratically at 256×256; diffusion models need large denoising-step datasets. The within-domain convolutional cGAN achieves PSNR 33.30 dB / SSIM 0.873 without overfitting, and Run 3 confirmed that additional capacity actively degrades water-index metrics — the bottleneck is domain data, not model expressivity.

---

## 🔄 End-to-End Pipeline

<!-- Replace with your pipeline diagram (Fig. 2 from paper) -->
![Pipeline](assets/fig2_pipeline.png)
*Full inference pipeline: Sentinel-1 SAR + pre-monsoon optical reference → 9-channel GAN input → reconstructed 6-band optical mosaic (72 patches/year) → parallel shoreline extraction (MNDWI threshold 0.0) and LULC classification (two-threshold MNDWI rule).*

---

## 📁 Repository Structure

```
.
├── gee/
│   └── patch_extraction.js        # GEE script: SAR + optical patch extraction
├── data/
│   ├── patches/                   # 256×256 px SAR–optical patch pairs (not tracked)
│   └── transects/                 # 119 DSAS transect shapefiles
├── model/
│   ├── generator.py               # Attention U-Net generator
│   ├── discriminator.py           # 70×70 PatchGAN discriminator
│   ├── loss.py                    # Adversarial + L1 composite loss
│   └── dataset.py                 # Patch dataset loader
├── train.py                       # Training script (100 epochs, seed 42)
├── inference.py                   # Monsoon inference + mosaicking
├── shoreline/
│   ├── extract_shoreline.py       # MNDWI threshold + Tampara Lake mask
│   └── dsas_transects.py          # DSAS transect intersection + signed distance
├── lulc/
│   └── classify_lulc.py           # Two-threshold MNDWI land cover classification
├── validation/
│   └── validate_ground_truth.py   # GAN vs. real Sentinel-2 comparison
├── checkpoints/
│   └── run2_unweighted/
│       └── gan_generator_epoch100.pt   # Primary model checkpoint
├── results/
│   ├── shoreline_distances.csv
│   ├── lulc_areas.csv
│   └── validation_summary.csv
├── assets/                        # Figures for README
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
git clone https://github.com/yourusername/monsoon-sar-gan-coastline.git
cd monsoon-sar-gan-coastline
pip install -r requirements.txt
```

**Requirements** (key packages):
```
torch>=2.0
torchvision
rasterio
geopandas
shapely
numpy
scikit-learn
matplotlib
```

---

## 🚀 Usage

### 1. Patch Extraction (Google Earth Engine)
Run `gee/patch_extraction.js` in the [GEE Code Editor](https://code.earthengine.google.com/). Set your GEE project to `sar-optical-synthesis`. Outputs 256×256 px patch pairs to Google Drive.

### 2. Training
```bash
python train.py \
  --data_dir data/patches \
  --epochs 100 \
  --lambda_l1 100 \
  --seed 42 \
  --checkpoint_dir checkpoints/run2_unweighted
```

### 3. Monsoon Inference
```bash
python inference.py \
  --checkpoint checkpoints/run2_unweighted/gan_generator_epoch100.pt \
  --sar_dir data/monsoon_sar/2023 \
  --ref_optical data/reference_optical/pre_monsoon.tif \
  --output_dir results/mosaics/2023
```

### 4. Shoreline Extraction & DSAS
```bash
python shoreline/extract_shoreline.py --mosaic results/mosaics/2023 --threshold 0.0
python shoreline/dsas_transects.py --shoreline results/shorelines/2023.shp
```

### 5. LULC Classification
```bash
python lulc/classify_lulc.py --mosaic results/mosaics/2023 --output results/lulc/2023
```

---

## 📊 Results & Figures

### GAN-Reconstructed Mosaics

<!-- Replace with Fig. 3 from paper -->
![Reconstructed Mosaics](assets/fig3_mosaics.png)
*GAN-reconstructed true-colour mosaics for 2021-06-03, 2022-06-10, 2023-06-05. 72 patches per mosaic at 10 m resolution. Note the markedly wider intertidal expression (pale strip) in 2023.*

### Per-Transect Seasonal Displacement

<!-- Replace with Fig. 4 from paper -->
![Seasonal Displacement](assets/fig4_displacement.png)
*Mean seasonal displacement (monsoon − post-monsoon) at 119 DSAS transects, south to north. Colour = LRR erosion class from the companion study. Negative values = landward displacement.*

### Validation Against Ground Truth

<!-- Replace with Fig. 5 from paper -->
![Validation](assets/fig5_validation.png)
*Left: GAN vs. ground-truth shoreline distance scatter (r = 0.689, n = 100). Right: Per-transect along-coast agreement profile for 2021 (green) and 2023 (red).*

### Monsoon LULC Maps

<!-- Replace with Fig. 6 from paper -->
![LULC Maps](assets/fig6_lulc.png)
*Monsoon-season MNDWI-threshold land cover: 2021, 2022, 2023. Brown = Land, orange = Intertidal, blue = Water. The 2023 map shows a visibly broader intertidal band.*

### DSAS Erosion Hotspot Overlay

<!-- Replace with Fig. 8 from paper -->
![Hotspot Overlay](assets/fig8_hotspot.png)
*Monsoon LULC with DSAS erosion class overlay. Red ▼ = High Erosion (LRR ≤ −1.0 m/yr, n=36); green ▲ = High Accretion (n=38); yellow ● = Stable (n=24). Mean LRR = −0.222 m/yr (companion study).*

---

## 📐 Validation Summary

| Metric | Value |
|--------|-------|
| Mean absolute error (MAE, m) | 81.8 |
| Std of absolute error (m) | 130.2 |
| Pearson r (GAN vs. Ground Truth) | 0.689 |
| % transects landward — GAN | **80.0%** |
| % transects landward — Ground Truth | **79.0%** |
| N matched transect-year observations | 100 |

**Fragmentation note:** 73.5% of transect intersections were MultiPoint geometries (shoreline crosses transect at multiple points). Standard post-processing (nearest-prior selection, Gaussian smoothing, morphological closing) all *increased* MAE rather than reducing it — confirming fragmentation is irreducible reconstruction noise at the transect level. The coastline-wide aggregate signal is robust; site-specific claims at individual transects are not warranted.

---

## 🔬 Data & Methods

- **SAR data:** Sentinel-1 IW GRD, DESCENDING orbit, 3 monsoon-onset dates (2021-06-03, 2022-06-10, 2023-06-05)
- **Optical reference:** Sentinel-2 L2A pre-monsoon clear-sky composite (Copernicus Programme / ESA)
- **Training data:** 311 paired patches, Brahmapur + Mahanadi delta AOIs; 70/15/15 split, seed 42; test n = 46 patches
- **Checkpoint:** `run2_unweighted/gan_generator_epoch100.pt`
- **Platform:** Google Earth Engine (project: `sar-optical-synthesis`) + Python (PyTorch, Rasterio, GeoPandas)
- **AOI:** 84.8433°E, 19.2221°N → 85.0599°E, 19.3906°N (EPSG:4326); UTM Zone 45N (EPSG:32645) for metric computation
- **MNDWI shoreline threshold:** 0.0 (Xu 2006), with diagonal coastal-strip mask to exclude Tampara Lake

---

## 📖 Citation

If you use this code or data, please cite:

```bibtex
@article{pradhan2025monsoon,
  title   = {Reconstructing Monsoon-Season Coastal Land Cover and Shoreline Position
             from Sentinel-1 SAR Using a Conditional Generative Adversarial Network:
             A Case Study of the Brahmapur Coastline, Odisha, India},
  author  = {Pradhan, Mohit},
  journal = {[Under Review]},
  year    = {2025},
  note    = {Companion study: Pradhan (2025a), multi-temporal DSAS analysis}
}
```

---

## 🙏 Acknowledgements

- Supervisor: **Prof. Ratnakar Dash**, NIT Rourkela
- **Google Earth Engine** cloud computing (project: `sar-optical-synthesis`)
- **Sentinel-1 / Sentinel-2** imagery: European Space Agency (ESA) Copernicus Programme
- **ERA5** reanalysis: Open-Meteo (https://open-meteo.com)

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<b>Department of Computer Science, NIST University, Brahmapur, Odisha, India</b><br>
mohit.pradhan.cse.2023@nist.edu
</div>