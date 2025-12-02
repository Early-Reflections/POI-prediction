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
```

### Data

Go to https://drive.google.com/drive/folders/1s5ps5Zk2932R3WRpNdNdekGHg0lOfB32 download the 'data' file into the root directory like:

Google drive gave me:
```
data-20251111T033806Z-1-002.zip  
data-20251111T033806Z-1-004.zip
traj2traj_pyg_data-001.pt
traj2traj_pyg_data-003.pt
```
The two traj2traj files are not the same byte count, so I don't know which is which.
Manually redownload the traj2traj files and placed them in the folders tky and tky\preprocessed.
The traj2traj file under tky/processed is the valid one.

```
/data
/dataset
/layer
...
```

### Option A) Local Python (venv; Python 3.10)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Option B) Docker Compose (CUDA 12.1; PyTorch 2.3.1)

Prereqs: NVIDIA drivers + NVIDIA Container Toolkit on host if using GPU.

```bash
docker compose build

# Train
docker compose run --rm app python run.py -f best_conf/{dataset}.yml
docker compose run --rm app python run.py -f best_conf/tky.yml

# Test
docker compose run --rm app python run_test.py -f best_conf/{dataset}.yml
docker compose run --rm app python run_test.py -f best_conf/tky.yml

# TensorBoard (training runs are under tensorboard/)
docker compose up -d tensorboard
# open http://localhost:6007

# To visualize run_test logs instead, run one-off TB on log/
docker compose run --service-ports --rm tensorboard \
  bash -lc "tensorboard --logdir log --port 6007 --host 0.0.0.0"
```

Notes:
- Compose binds the repo at /app, so local data/ is available in the container.
- GPU usage: compose is set to NVIDIA GPU id 0 for the app service; adjust `device_ids` if needed.


## Main dependencies:

- torch==2.3.1
- torch-geometric==2.5.3
- transformers==4.41.2

## Run

Train/validate/test per YAML (logs in: tensorboard/, log/).

```bash
python run.py -f best_conf/{dataset_name}.yml
# dataset_name ∈ {nyc, tky, ca}
```

Copy paste example:
```bash
python run.py -f best_conf/nyc.yml
```

Open TensorBoard; run_test.py writes to log/, run.py writes to tensorboard/.

```bash
tensorboard --logdir log
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

### Appendix

sudo chown -R $USER:$USER log
