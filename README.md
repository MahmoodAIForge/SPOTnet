# SPOTnet: Weakly-Supervised Specular Reflection Detection and Removal for Endoscopic Images

<p align="center">
  <img src="assets/pipeline.png" width="800"/>
</p>

---

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

SPOTnet achieves **state-of-the-art** performance with only **0.77M parameters** — **121× smaller** than EndoSRR (93.79M) while outperforming it on IoU (+8.3%) and Recall (+9.8%), all without any manual annotations.

---

## Inpainting

We use **LaMa** (Large Mask inpainting) for specular removal:
```bash
pip install simple-lama-inpainting
```

The iterative pipeline runs up to 5 detect→inpaint cycles until the specular ratio drops below 0.015%.

---

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
