# Context-aware Spatiotemporal Graph Attention Network for Next POI Recommendation

This repository provides the implementation of our paper
“Context-aware Spatiotemporal Graph Attention Network for Next POI Recommendation” (Han et al., KSEM 2025).

The model introduces graph attention and channel attention mechanisms to adaptively balance spatial and temporal features in next Point-of-Interest (POI) recommendation tasks.

## Overview

- Extends the original STHGCN framework with context-aware attention.
- Enhances representation of spatio-temporal dependencies among users, POIs, and trajectories.
- Fully compatible with the same dataset format and training scripts as the original STHGCN.

## Setup

```bash
git clone https://github.com/yourusername/POI-prediction.git

cd POI-prediction

pip install -r requirements.txt
```

Go to https://drive.google.com/drive/folders/1s5ps5Zk2932R3WRpNdNdekGHg0lOfB32 download the 'data' file into the root directory like:

```
/data
/dataset
/layer
...
```


## Main dependencies:

- torch==2.3.1
- torch-geometric==2.5.3
- transformers==4.41.2

## 🚀 Run

### **1. Train the model**

```bash
python run.py -f best_conf/{dataset_name}.yml
# dataset_name ∈ {nyc, tky, ca}
```

### **2. Test the model**

1. change the test.yml file, load the checkpoint like:
D:/Projects/EarlyRef/POI-prediction/tensorboard/20251015_142454/nyc/

2. run in terminal
```bash
python run_test.py -f best_conf/{dataset_name}_test.yml
# dataset_name ∈ {nyc, tky, ca}
```
this will generate a test_prediction_top20.csv file.

## Visulization
use the test_prediction_top20.csv file and the sample file

run select_traj.py

this will generate several html to show the prediction steps.


## Citation

If you use this repository, please cite:

Han Qiuhan, Wang Qian, Yoshikawa Atsushi, and Yamamura Masayuki.
Context-aware Spatiotemporal Graph Attention Network for Next POI Recommendation.
Proceedings of KSEM 2025.