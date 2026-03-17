# Objective 1B: U-Net DEM-Based Active Fault Detection

This notebook implements a U-Net segmentation model to detect active faults from LiDAR-derived Digital Elevation Model (DEM) data, as part of the Prithvi-EO 2.0 Fault Detection project.

## Study Area
**Parkfield, CA** (lat 35.27, lon -119.82) — San Andreas Fault zone

## Model
- **Architecture**: U-Net with ResNet34 encoder
- **Pretrained weights**: ImageNet
- **Input**: 3-channel DEM — `[hillshade, slope, hillshade]`
- **Output**: Binary segmentation mask (fault / no-fault)
- **Parameters**: ~24.4M (fully trainable)

## Data

### Input Data
| File | Description | Resolution |
|------|-------------|------------|
| `viz.be_hillshade.tif` | LiDAR-derived hillshade | 0.5m |
| `viz.be_slope.tif` | LiDAR-derived slope | 0.5m |

Both files were provided from the Parkfield study area. Hillshade is duplicated as the 3rd channel to satisfy the ImageNet pretrained ResNet34 input format.

### Ground Truth Labels
- Source: **USGS Quaternary Fault and Fold Database**
- Processing: Fault lines rasterized at 0.5m resolution with a **10m buffer**
- Stored as binary masks (1 = fault, 0 = background)

### Dataset Split
| Split | Total patches | Fault patches | Fault pixel % |
|-------|--------------|---------------|---------------|
| Train | 2,733 | 159 (6%) | 1.20% |
| Val   | 585  | 34 (6%)  | 1.39% |
| Test  | 588  | 35 (6%)  | 1.15% |
| **Total** | **3,906** | **228 (5.8%)** | — |

Patch size: **128 × 128 pixels** (64m × 64m)

## Results

Test set evaluation (optimal threshold = 0.35):

| Metric | Value |
|--------|-------|
| mIoU | **0.688** |
| IoU_fault | **0.385** |
| F1 | **0.556** |
| Accuracy | 0.992 |

## How to Run

### Prerequisites
```
Google Colab (A100 GPU recommended)
Google Drive with the following structure:

MyDrive/prithvi_fault/
├── data/
│   └── patches/
│       └── parkfield_dem/
│           ├── train.npz
│           ├── val.npz
│           └── test.npz
└── checkpoints_unet/   (created automatically)
```

### Steps

1. **Open** `unet_dem_fault.ipynb` in Google Colab

2. **Set runtime** to A100 GPU:
   `Runtime > Change runtime type > A100`

3. **Run Cell 1** — installs `segmentation-models-pytorch`

4. **Run Cell 2** — mounts Google Drive and verifies data

5. **Run Cell 3** — creates Dataset and DataLoader

6. **Run Cell 4** — builds U-Net model, loss, and optimizer

7. **Run Cell 5** — trains the model (~3 min on A100, early stopping at ~62 epochs)

8. **Run Cell 6** — evaluates on test set with threshold tuning

9. **Run Cell 7** — generates training curves and prediction visualizations

### Output Files
After running all cells, the following files are saved to Google Drive:

```
MyDrive/prithvi_fault/
├── unet_training_curve.png      # Train loss + validation metrics
├── unet_test_predictions.png    # Hillshade / GT / Pred / Overlay
└── checkpoints_unet/
    ├── best_unet.pth            # Best model checkpoint
    ├── history.json             # Training history
    └── test_results.json        # Final test metrics
```

## Expected Output

### Training (Cell 5)
```
============================================================
U-Net DEM Fault Detection
============================================================
Ep   1 | Loss:1.3979 | mIoU:0.4904 | IoU_fault:0.0119 | F1:0.0214
...
Ep  42 | Loss:0.1187 | mIoU:0.5206 | IoU_fault:0.0655 | F1:0.0675
Early stopping at epoch 62
Done in 3.2 min | Best IoU_fault: 0.0655
```

### Test Evaluation (Cell 6)
```
Optimal threshold: 0.35 (val IoU_fault=0.4010)

==================================================
TEST RESULTS (threshold=0.35)
==================================================
  mIoU      : 0.6883
  IoU_fault : 0.3853
  F1        : 0.5562
  Accuracy  : 0.9915
```

### Visualization (Cell 7)
- **Left**: Training loss curve decreasing from ~1.4 to ~0.07
- **Right**: Validation IoU_fault peaking around epoch 42
- **Prediction grid**: 6 fault-containing test patches showing hillshade input, ground truth mask, predicted probability map, and overlay

## Notes
- The model uses **WeightedRandomSampler** (15x oversampling for fault patches) to handle the ~5% fault pixel imbalance
- **Threshold tuning** on the validation set (range 0.05–0.50) shows the model predicts fault regions with relatively low probability, so threshold=0.35 outperforms the default threshold=0.50
- Ground truth masks cover a 10m buffer around the USGS fault traces, which at 0.5m resolution results in ~20-pixel wide fault bands
