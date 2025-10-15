### Context-aware Spatiotemporal Graph Attention Network for Next POI Recommendation

This repository provides the implementation of our paper
“Context-aware Spatiotemporal Graph Attention Network for Next POI Recommendation” (Han et al., KSEM 2025).

The model introduces graph attention and channel attention mechanisms to adaptively balance spatial and temporal features in next Point-of-Interest (POI) recommendation tasks.

## Overview

Extends the original STHGCN framework with context-aware attention.

Enhances representation of spatio-temporal dependencies among users, POIs, and trajectories.

Fully compatible with the same dataset format and training scripts as the original STHGCN.

### Setup
git clone https://github.com/yourusername/POI-prediction.git
cd POI-prediction
pip install -r requirements.txt


# Main dependencies:

torch==2.3.1
torch-geometric==2.5.3
transformers==4.41.2

## Run
python run.py -f best_conf/{dataset_name}.yml
# dataset_name ∈ {nyc, tky, ca}

## Citation

If you use this repository, please cite:

Han Qiuhan, Wang Qian, Yoshikawa Atsushi, and Yamamura Masayuki.
Context-aware Spatiotemporal Graph Attention Network for Next POI Recommendation.
Proceedings of KSEM 2025.