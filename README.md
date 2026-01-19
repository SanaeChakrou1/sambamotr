# SambaMOTR

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

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
- Python 3.8 or higher
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
**Alternative:** Use the automated setup script:
```bash
bash scripts/setup_all.sh
```
## ⚡ Quick Start

### Run OSNet Tracking
```bash
cd osnet
python main.py \
    --mode submit \
    --config-path ../configs/osnet_config.yaml \
    --submit-model ../checkpoints/sambamotr_dab_dancetrack.pth \
    --dataset DanceTrack \
    --data-root ../data/dancetrack/ \
    --submit-data-split val \
    --use-reid \
    --reid-thresh 0.7
```

### Run DeepSORT Tracking
```bash
cd deepsort
python main.py \
    --mode submit \
    --config-path ../configs/deepsort_config.yaml \
    --submit-model ../checkpoints/sambamotr_dab_dancetrack.pth \
    --dataset DanceTrack \
    --data-root ../data/dancetrack/ \
    --submit-data-split val \
    --use-reid
```
