# SambaMOTR

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A comparative study of two different multi-object tracking approaches: **OSNet-based tracking** and **DeepSORT tracking**.


## Overview

This repository provides implementations and comparisons of two distinct tracking methods:

**OSNet-based Tracking**
- Uses OSNet (Omni-Scale Network) for appearance-based re-identification
- Extracts discriminative appearance features
- Focuses on visual similarity for identity matching

### **DeepSORT Tracking**
- Multi-object tracking with Kalman filtering
- Combines motion prediction with appearance features
- Handles occlusions and ID switches effectively

## Purpose

This project allows you to:
- Compare the performance of OSNet and DeepSORT on the same dataset
- Evaluate different tracking strategies (appearance-only vs. motion+appearance)
- Understand the strengths and weaknesses of each approach
- Choose the best method for your specific use case

## Features

-  **Two Independent Implementations**: OSNet and DeepSORT as separate modules
-  **DanceTrack Support**: Tested on challenging dance tracking scenarios
-  **Easy Comparison**: Run both methods with similar parameters
-  **GPU Accelerated**: Fast inference with CUDA support
-  **Detailed Metrics**: MOTA, IDF1, HOTA for comprehensive evaluation

## Installation

### Prerequisites
- Python 3.11 or higher
- CUDA 11.0+ (for GPU support)
- 8GB+ RAM recommended

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/mattiasegu/sambamotr.git
cd sambamotr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
### Download Dataset (DanceTrack)
```bash
# Navigate to data folder
cd data

# Clone DanceTrack dataset
git clone https://huggingface.co/datasets/noahcao/dancetrack
cd dancetrack
### Download Model Weights
```bash
# Download pretrained checkpoint
cd checkpoints
wget https://huggingface.co/mattiasegu/sambamotr/resolve/main/sambamotr_pretrained/dancetrack/dab_detr/sambamotr_dab_dancetrack.pth
cd ..
```


##  Quick Start

### Run OSNet Tracking
```bash
cd osnet
!python /content/sambamotr/main.py \
    --mode submit \
    --config-path /content/sambamotr/output/eval_results/train/config.yaml \
    --available-gpus "0" \
    --submit-model /content/sambamotr/checkpoints/sambamotr_dab_dancetrack.pth \
    --dataset DanceTrack \
    --data-root data/ \
    --submit-data-split val \
    --exp-name osnet_eval \
    --submit-dir /content/sambamotr/output/eval_results \
    --batch-size 1 \
    --use-reid \
    --reid-thresh 0.7
```

### Run DeepSORT Tracking
```bash
cd deepsort
!python main.py \
    --mode eval \
    --config-path configs/sambamotr/dancetrack/def_detr/train_residual_masking_sync_longer.yaml \
    --available-gpus "0" \
    --eval-model /content/sambamotr/checkpoints/sambamotr_dab_dancetrack.pth \
    --dataset DanceTrack \
    --data-root data/\
    --eval-data-split val \
    --eval-dir /content/sambamotr/output/eval_results \
    --outputs-dir /content/sambamotr/output/eval_results\
    --batch-size 1
```
