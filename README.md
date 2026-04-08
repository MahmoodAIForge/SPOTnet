# SPOTnet: Weakly-Supervised Specular Reflection Detection and Removal for Endoscopic Images


https://github.com/user-attachments/assets/cdcb437a-ee68-43e8-965e-383ce04ed900

## Overview

**SPOTnet** is a weakly-supervised deep learning pipeline for detecting and removing specular reflections (SR) in endoscopic images. It requires **no manual annotations** — instead, a real specular patch library is automatically built from HSV-based detection and used to train a lightweight UNet detector.

The full pipeline:
```
Endoscopic Frames
      │
      ▼
┌─────────────────────────────────┐
│  Step 1: Specular Library       │  HSV detection → Quality filtering → Patch library
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  Step 2-5: SPOTnet Training     │  UNet + Multi-Scale + Boundary Loss → SPOTnet
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  Step 6: Iterative Inpainting   │  SPOTnet + LaMa → Specular-Free Output 
└─────────────────────────────────┘
```
<img width="1992" height="1586" alt="Pipeline" src="https://github.com/user-attachments/assets/ca130335-287d-4893-9506-f2a04a05464c" />

---

## Results

| Method | Accuracy | IoU | Precision | Recall | E-measure |
|--------|----------|-----|-----------|--------|-----------|
| Arnold et al. | 0.9620 | 0.3698 | 0.5361 | 0.5885 | 0.5307 |
| **SPOTnet (Ours)** | **0.9657** | **0.3798** | 0.5419 | **0.6016** | 0.5371 |

*Evaluated on EndoRR test set. SPOTnet outperforms Arnold et al. on IoU (+2.7%) and Recall (+2.2%) without any manual annotations.*

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_specular_library_creation.ipynb` | Extract specular patches using HSV detection |
| 02 | `02_spotnet_baseline.ipynb` | Train baseline UNet (BCE + Dice) |
| 03 | `03_spotnet_multiscale.ipynb` | Add multi-scale training |
| 04 | `04_spotnet_boundary.ipynb` | Add boundary-aware loss |
| 05 | `05_spotnet_final.ipynb` | Full SPOTnet (MS + Boundary) |
| 06 | `06_iterative_inpainting.ipynb` | Detect + inpaint pipeline |
| 07 | `07_benchmark.ipynb` | Evaluate all models vs Arnold baseline |

---

## Installation

```bash
git clone https://github.com/MahmoodAIForge/SPOTnet.git
cd SPOTnet
pip install -r requirements.txt
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+

```bash
pip install -r requirements.txt
```

---

## Dataset Structure

```
data/
├── frames/              ← endoscopic frames for library creation
├── real_specular_library_v1/
│   ├── patches/
│   ├── masks/
│   └── meta/
└── dataset/
    └── EndoRR/
        └── test/
            ├── image/
            └── mask/
```

---

## Quick Start

**1. Build the specular library:**
```bash
jupyter nbconvert --to notebook --execute 01_specular_library_creation.ipynb
```

**2. Train SPOTnet:**
```bash
jupyter nbconvert --to notebook --execute 05_spotnet_final.ipynb
```

**3. Run inpainting:**
```bash
jupyter nbconvert --to notebook --execute 06_iterative_inpainting.ipynb
```

---

## Model Architecture

SPOTnet uses a lightweight **UNet** with:
- 3-level encoder-decoder with skip connections
- SiLU activations + BatchNorm
- Only **~500K parameters**
- Trained on 128×128 patches, runs on full-resolution images via **tiled inference**

---

## Ablation Study

| Configuration | Multi-Scale | Boundary Loss | IoU | Recall | E-measure |
|---------------|------------|---------------|-----|--------|-----------|
| Baseline | ✗ | ✗ | 0.594 | 0.660 | 0.938 |
| + Multi-Scale | ✓ | ✗ | 0.575 | 0.615 | 0.908 |
| + Boundary Loss | ✗ | ✓ | 0.581 | 0.625 | 0.915 |
| **SPOTnet** | **✓** | **✓** | **0.613** | **0.740** | **0.963** |

---

## Results

### Specular Reflection Detection

| Method | Params | Time (ms) | Accuracy | IoU | Recall | E-measure |
|--------|--------|-----------|----------|-----|--------|-----------|
| Arnold et al. | N/A | ~15 | 0.993±0.005 | 0.392±0.112 | 0.402±0.112 | 0.732±0.112 |
| EndoSRR | 93.79M | ~150 | 0.994±0.004 | 0.566±0.150 | 0.674±0.157 | 0.959±0.055 |
| **SPOTnet (Ours)** | **0.77M** | **~50** | **0.995±0.003** | **0.613±0.116** | **0.740±0.106** | **0.963±0.041** |

Evaluated on **TEMSET-24k** test set. SPOTnet achieves **state-of-the-art** performance with only **0.77M parameters** — **121× smaller** than EndoSRR (93.79M) while outperforming it on IoU (+8.3%) and Recall (+9.8%), all without any manual annotations.

---
<img width="3266" height="1222" alt="detection_comparison (1)" src="https://github.com/user-attachments/assets/5eebfd83-bb4b-449e-935d-3ceffb763181" />
<p><em>Qualitative comparison of specular reflection detection method</em></p>


## Inpainting

We use **LaMa** (Large Mask inpainting) for specular removal:
```bash
pip install simple-lama-inpainting
```

The iterative pipeline runs up to 5 detect→inpaint cycles until the specular ratio drops below 0.015%.

---
## Qualitative Results

<img width="1628" height="1172" alt="GeneralisablityTest (1)" src="https://github.com/user-attachments/assets/0f7034bb-ff40-4713-a502-52711896bfcd" />
<p><em>Qualitative results of the proposed specular detection and inpainting pipeline on diverse surgical datasets.</em></p>

## Downstream Segmentation Validation

To evaluate the impact of specular reflection removal on downstream semantic segmentation, we trained **DeepLabV3+** separately on the original expert-labelled **TEMSET-24K** images and on the corresponding **specular-free enhanced images** produced by our pipeline. Training on the enhanced images led to consistent improvements across **11 semantic classes**, with the **average validation F1 score increasing from 0.713 to 0.905**.

| Class | Train Acc (Original) | Train Acc (Enhanced) | Val Acc (Original) | Val Acc (Enhanced) | Train F1 (Original) | Train F1 (Enhanced) | Val F1 (Original) | Val F1 (Enhanced) |
|------|-----------------------|----------------------|--------------------|--------------------|---------------------|---------------------|-------------------|-------------------|
| **Anatomy** |||||||||
| RectalWall | 0.947 | 0.986 | 0.950 | 0.977 | 0.894 | 0.971 | 0.892 | 0.951 |
| Lesion | 0.986 | 0.997 | 0.988 | 0.996 | 0.866 | 0.974 | 0.867 | 0.953 |
| RectalWallDefect | 0.976 | 0.994 | 0.978 | 0.990 | 0.835 | 0.957 | 0.834 | 0.927 |
| **Instruments** |||||||||
| Rectoscope | 0.987 | 0.997 | 0.987 | 0.996 | 0.885 | 0.969 | 0.886 | 0.960 |
| Retractor | 0.992 | 0.998 | 0.993 | 0.997 | 0.872 | 0.972 | 0.875 | 0.956 |
| Diathermy | 0.993 | 0.999 | 0.994 | 0.998 | 0.823 | 0.972 | 0.832 | 0.957 |
| **Miscellaneous** |||||||||
| Landmark | 0.995 | 0.998 | 0.996 | 0.998 | 0.431 | 0.831 | 0.460 | 0.804 |
| Blood | 0.998 | 1.000 | 0.998 | 0.999 | 0.387 | 0.916 | 0.351 | 0.679 |
| Feces | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.892 | 0.000 | 0.871 |
| PoorVisibility | 0.993 | 0.999 | 0.992 | 0.995 | 0.894 | 0.982 | 0.854 | 0.899 |
| OutOfBody | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 | 0.999 | 0.996 | 0.996 |
| **Average** | **0.988** | **0.997** | **0.989** | **0.995** | **0.717** | **0.949** | **0.713** | **0.905** |

<p><em>Impact of specular removal on downstream semantic segmentation using DeepLabV3+ on TEMSET-24K.</em></p>


## Citation

If you use this work, please cite:
```bibtex
@article{spotnet2025,
  title   = {SPOTnet: Weakly-Supervised Specular Reflection Detection and Removal for Minimal Access Surgery},
  author  = {},
  journal = {},
  year    = {2025}
}
```

---

## Acknowledgements

- [LaMa](https://github.com/saic-mdal/lama) for inpainting
- [EndoSRR](https://github.com/Tobyzai/EndoSRR) for dataset and evaluation protocols
- [Arnold et al.](https://doi.org/10.1007/s11548-010-0507-y) for the HSV baseline
