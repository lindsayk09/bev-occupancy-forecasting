# Temporal BEV Occupancy Forecasting
### ConvLSTM on Waymo Open Dataset v2.0.1

Predicts future Bird's-Eye View occupancy grids from sequences of past camera
frames — a core task in autonomous driving scene understanding and a direct
precursor to full **Semantic 4D Occupancy Forecasting**.

This project builds directly on [Semantic-Feature-Visualisation-Waymo-Open-Dataset](https://github.com/lindsayk09/Semantic-Feature-Visualisation-Waymo-Open-Dataset),
reusing the same camera frames extracted there, and extends the work from
*understanding a single scene* to *predicting how that scene evolves over time*.

---

## Overview

Occupancy forecasting asks a deceptively simple question: given what a self-driving
car has seen in the last few moments, what will the space around it look like one
moment from now? Answering this well, without relying on expensive dense 3D
annotations, is one of the open problems behind current autonomous driving research.

This project tackles a simplified but representative version of that problem.
Camera frames from a Waymo driving sequence are converted into top-down occupancy
grids, and a Convolutional LSTM is trained to predict the next grid in the
sequence from the three grids that came before it. The model is evaluated using
**IoU (Intersection over Union)**, the standard metric for occupancy prediction
benchmarks such as nuScenes and OpenOccupancy.

The pipeline runs end to end on a CPU laptop in under five minutes, using only
the 45 camera frames already extracted for the companion DINOv2 project.

---

## Method

The pipeline has three stages:

**1. Camera → BEV via Inverse Perspective Mapping.**
A homography transform "unfolds" the front-facing camera image into a top-down
view. Canny edge detection on the warped image highlights road markings, lane
boundaries, vehicles, and roadside structures, producing a binary occupancy grid.

**2. Temporal encoding via ConvLSTM.**
A 2-layer Convolutional LSTM processes a sequence of 3 past BEV grids. Unlike a
standard LSTM, ConvLSTM replaces the internal gate computations with
convolutions, preserving the spatial layout of the grid while still modelling
how it changes over time.

**3. Occupancy prediction.**
A convolutional decoder head takes the ConvLSTM's final hidden state and
predicts the next occupancy grid as a probability map, thresholded at 0.5 to
produce a binary prediction.

Waymo camera frames  (PNG sequence, FRONT camera)

|

Inverse Perspective Mapping (OpenCV homography)

|

Binary BEV occupancy grids  [128 x 128]

|

ConvLSTM encoder  (2 layers, 32 hidden channels)

Input: [B, T=3, 1, 128, 128]

|

Final hidden state  [B, 32, 128, 128]

|

Conv decoder head  (3 layers)

|

Predicted occupancy map  [B, 1, 128, 128]  in [0,1]

|

Threshold @ 0.5 -> Binary prediction

|

IoU evaluation vs ground truth

---

## Results

Trained for 20 epochs on 42 sequence samples (34 train / 8 validation) derived
from 45 camera frames across 3 Waymo driving segments.

| Metric | Value |
|---|---|
| Best validation IoU | **0.728** |
| Final train IoU | 0.791 |
| Final train loss (BCE) | 0.370 |
| Final val loss (BCE) | 0.503 |
| Model parameters | 125,889 |
| Per-sample IoU range (validation) | 0.577 - 0.890 |

**Camera to BEV conversion (Inverse Perspective Mapping):**

![BEV IPM sample](outputs/bev_ipm_sample.png)

The IPM transform correctly unfolds the road's perspective into a flat top-down
view. Road curvature, lane markings, and roadside structures remain spatially
coherent in the warped grid.

**Training curves:**

<img width="1806" height="643" alt="training_curves" src="https://github.com/user-attachments/assets/a9d2f4cb-b2d8-426b-82f1-3519b60be412" />


Loss decreases smoothly across training with no sign of instability. IoU rises
from 0.60 at epoch 1 to a validation plateau around 0.73 after epoch 10, with
training IoU continuing to climb to 0.79. The modest gap between train and
validation IoU is expected given the small dataset (42 samples) and indicates
the model is learning genuine spatial-temporal structure rather than memorising
the training sequences outright.

**Predictions vs ground truth:**

<img width="2327" height="2120" alt="predictions" src="https://github.com/user-attachments/assets/3059d92d-5f8f-4ad9-9d9f-506f525d7757" />


Each row shows three input timesteps, the ground truth at T+1, and the model's
predicted occupancy probability map. Per-sample IoU ranges from 0.577 to 0.890
across the four examples shown. The predicted heatmaps closely track the
spatial layout of occupied regions in the ground truth, including road
boundaries and structural edges that persist across frames.

---

## Connection to Semantic 4D Occupancy Forecasting

This project addresses the **temporal prediction** component of 4D occupancy
forecasting in isolation. The full research direction extends this work in two
ways:

1. **Semantic labels.** Instead of binary occupancy, predict per-voxel class
   labels (road, vehicle, pedestrian, vegetation). This is where the companion
   DINOv2 project becomes relevant: its patch-level semantic features are
   exactly the kind of signal that would be fused into the BEV representation
   to move from "is this space occupied" to "what is occupying this space."

2. **Full 3D voxel representation.** Extend the 2D BEV grid used here to true
   3D voxels, using depth estimation or LiDAR fusion, matching the output
   space of models such as OccFormer and UniOcc.

Together, the two projects in this portfolio cover both pillars of
weakly-supervised 4D occupancy forecasting: **semantic feature extraction**
(DINOv2 project) and **temporal occupancy prediction** (this project).

---

## Dataset

**Waymo Open Dataset v2.0.1** — the same 3 validation segments used in the
companion DINOv2 project:

| Segment ID | Scene type |
|---|---|
| `10203656353524179475_7625_000_7645_000` | Highway / construction zone |
| `1024360143612057520_3580_000_3600_000` | Urban intersection / pedestrians |
| `10247954040621004675_2180_000_2200_000` | Residential street / vehicles |

Camera: FRONT (camera ID 1). 15 frames extracted per segment, 45 total.

---

## Setup

```bash
git clone https://github.com/lindsayk09/bev-occupancy-forecasting.git
cd bev-occupancy-forecasting
pip install -r requirements.txt
```

**Run the full pipeline** (BEV generation, training, and evaluation in one step):

```bash
cd src
python train_evaluate.py \
    --frames_dir ./data/frames \
    --out_dir    ../outputs \
    --epochs     20 \
    --grid_size  128
```

**Generate BEV grids only**, to inspect the IPM output without training:

```bash
python bev_generator.py \
    --frames_dir ./data/frames \
    --out_dir    ../outputs/bev_grids \
    --visualise
```

---

## Project structure

bev-occupancy/

├── src/

│   ├── train_evaluate.py    Full pipeline: BEV generation -> training -> evaluation

│   ├── convlstm_model.py    ConvLSTM architecture, dataset, and IoU metric

│   └── bev_generator.py     IPM-based camera -> BEV grid conversion

├── outputs/

│   ├── bev_grids/            Binary .npy occupancy grids (intermediate)

│   ├── bev_ipm_sample.png    IPM visualisation

│   ├── training_curves.png   Loss and IoU curves

│   ├── predictions.png       Predicted vs ground truth

│   ├── model_best.pth        Best model checkpoint (by val IoU)

│   └── results.txt           Final IoU score summary

├── data/                     Waymo camera frames (not tracked in git)

└── requirements.txt

---

## References

- Shi et al., *Convolutional LSTM Network: A Machine Learning Approach for
  Precipitation Nowcasting*, NeurIPS 2015
- Zhang et al., *OccFormer: Dual-path Transformer for Vision-based 3D Semantic
  Occupancy Prediction*, ICCV 2023
- Wang et al., *OpenOccupancy: A Large Scale Benchmark for Surrounding Semantic
  Occupancy Perception*, ICCV 2023
- Sun et al., *Scalability in Perception for Autonomous Driving: Waymo Open
  Dataset*, CVPR 2020
