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
