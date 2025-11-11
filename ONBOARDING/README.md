# POI-prediction Onboarding

This document is the source of truth for understanding, setting up, and extending the repository.

- Paper: Context-aware Spatiotemporal Graph Attention Network for Next POI Recommendation (Han et al., KSEM 2025)
- Core idea: A hypergraph-based GNN with context-aware attention that models spatiotemporal dependencies among users, POIs, check-ins, and trajectories.


## Repository structure

- conf/
  - best_conf/*.yml
  - ablation_conf/*.yml
  - README.md (all configuration fields explained)
- data/
  - <dataset>/{raw, preprocessed}
  - Preprocessed files are generated or provided via the drive link
- dataset/
  - lbsn_dataset.py (PyG tensors and labels loader)
- layer/
  - conv.py (HypergraphTransformer)
  - sampler.py (NeighborSampler dataloader)
  - embedding_layer.py, bert_embedding_layer.py, st_encoder.py
- metric/
  - rank_metric.py (Recall@K, NDCG@K, MAP@K, MRR, per-class acc)
- model/
  - sthgcn.py (main model)
  - seq_transformer.py (ablation model)
  - attention.py (experimental/alternate modules)
- preprocess/
  - preprocess_main.py (entry), generate_hypergraph.py, file_reader.py, preprocess_fn.py
- utils/
  - sys_util.py (root detection, logging, seeding, visualization), pipeline_util.py (eval + checkpoint), conf_util.py
- Entry scripts
  - run.py (train/validate/test)
  - run_test.py (evaluation/visualization utilities)
  - run_cold_start.py (cold-start splits and eval helpers)
  - multiple_run.py (batch experiments; Windows-style env command inside)
  - compute_acc.py (parse logs and aggregate metrics)


## Data expectations and preprocessing

- Datasets: nyc, tky, ca
- Raw files expected under data/<dataset>/raw
  - nyc: NYC_train.csv, NYC_val.csv, NYC_test.csv
  - tky: dataset_TSMC2014_TKY.txt
  - ca:  dataset_gowalla_ca_ne.csv
- Preprocessing will create data/<dataset>/preprocessed with at least:
  - sample.csv, train_sample.csv, validate_sample.csv, test_sample.csv
  - ci2traj_pyg_data.pt (check-in to trajectory incidence and features)
  - traj2traj_pyg_data.pt (trajectory-to-trajectory dynamic edges/features)
- Precomputed artifacts
  - bert_address_embedding.pt must exist at data/<dataset>/preprocessed
    - Used by bert_embedding_layer.BertCheckinEmbedding
    - If missing: training will fail at model init. You can provide the file from the Google Drive bundle or generate it with sentence_encoder.py (update its paths before running).
- Google Drive bundle
  - Place the unpacked content under project root data/ so that each dataset folder contains the required preprocessed files.


## Configuration system

- Pass config relative to conf/ (do NOT prefix with conf/). Example: -f best_conf/nyc.yml
- utils/conf_util.Cfg loads: conf/<yaml>
- Key groups (see conf/README.md for full reference):
  - dataset_args: dataset_name, session_time_interval, group_type, thresholds, slots, etc.
  - model_args: model_name, sizes, dropout, embed sizes, encoders, do_traj2traj, etc.
  - conv_args: hypergraph transformer settings (heads, fusion modes, residuals, dropout, slopes, feature attention)
  - run_args: seed, gpu, batch sizes, learning rate, steps/epochs, workers, logging/ckpt configs, visualize
  - seq_transformer_args: ablation model hyperparams

Important: model_args.sizes is a dash-separated string (e.g., "300-500") and is parsed into a list. The last element is the ci2traj sample size; preceding elements are multi-hop traj2traj sample sizes.


## Environment and installation

- Python: 3.10+ is known-good.
- requirements.txt is pinned for PyTorch 2.3.1 and PyG 2.5.3 family.
- GPU by default
  - requirements.txt includes options to pull CUDA 12.1 wheels
    - --extra-index-url https://download.pytorch.org/whl/cu121
    - --find-links https://data.pyg.org/whl/torch-2.3.1+cu121.html
  - Install:
    - python3 -m venv .venv
    - source .venv/bin/activate
    - pip install -r requirements.txt
  - Verify:
    - python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
- CPU alternative (if needed)
  - Replace the two pip option lines with CPU equivalents:
    - --extra-index-url https://download.pytorch.org/whl/cpu
    - --find-links https://data.pyg.org/whl/torch-2.3.1+cpu.html


## Quickstart

- Prepare data:
  - Ensure data/<dataset>/preprocessed contains the required .csv and .pt files, including bert_address_embedding.pt
- Run training + eval + tensorboard logging:
  - python run.py -f best_conf/{dataset}.yml
  - Example: python run.py -f best_conf/nyc.yml
  - Outputs:
    - tensorboard/<timestamp>/<dataset>/ (checkpoint.pt stored here)
    - log/<timestamp>/<dataset>/train.log
- Evaluate a saved checkpoint only (alternate utilities in run_test.py also available):
  - Set run_args.do_train: False, do_test: True, and init_checkpoint: path to tensorboard/<timestamp>/<dataset>


## Model overview (STHGCN)

- Input graph (PyG-like): union of
  - traj2traj: dynamic relations between trajectories (intra-user and inter-user), with edge_attr (source_size, target_size, similarity), edge_type, edge_delta_t, edge_delta_s
  - ci2traj: check-in to trajectory incidence, with edge_t (check-in time), edge_delta_t (time to trajectory end), edge_delta_s (distance to trajectory end)
- Embedding layers
  - Check-in features fuse: user, poi, category, weekday, hour, and address BERT embedding (reduced to embed_size)
  - Optional edge embeddings for type/attr depending on fusion modes
- Encoders
  - Time: TimeEncoder
  - Distance: DistanceEncoderSTAN/TimeEncoder/HSTLSTM/Simple, configured via model_args.distance_encoder_type
- Convolutions
  - conv_for_time_filter: HypergraphTransformer over ci2traj with time/distance conditioning
  - Optional multi-layer traj2traj HypergraphTransformer stack if do_traj2traj=True
- Output
  - Linear layer to #POIs; loss=CrossEntropy over next POI id
- Training loop
  - Built around NeighborSampler producing batched subgraphs and labels
  - accelerate.Accelerator for device/backward
  - Metrics: Recall@{1,5,10,20}, NDCG@{1,5,10,20}, MAP@{1,5,10,20}, MRR; tensorboard summaries


## Data loader (NeighborSampler) semantics

- Sizes: [traj2traj hop1, traj2traj hop2, ..., ci2traj]
- Returns adjs in order: [filtered_ci2traj, ci2traj, traj2traj]
- Applies temporal filtering to ci2traj (keep check-ins before target time)
- Filters traj2traj by Jaccard thresholds (intra/inter) and leakage rules
- Generates target trajectory features (size, mean lon/lat/time, min/max time) per batch
- Populates labels y[:, 0] as POI ids for loss/metrics


## Metrics and logging

- Logs to log/<timestamp>/<dataset>/train.log (or test.log)
- TensorBoard at tensorboard/<timestamp>/<dataset>/
  - Scalars: train/loss_step, test/Recall@K, validate/*, etc.
  - HParams: saved at the end of test
- Checkpoints
  - Saved to tensorboard/<timestamp>/<dataset>/checkpoint.pt


## Common pitfalls and fixes

- Config path: pass -f best_conf/nyc.yml (NOT conf/best_conf/nyc.yml) due to conf_util joining with conf/ internally.
- Missing bert_address_embedding.pt: place it into data/<dataset>/preprocessed or generate it (edit sentence_encoder.py paths; it assumes Windows drive in its defaults).
- Dataset path/layout: ensure raw and preprocessed exist under data/<dataset>/ and names match expected files.
- NeighborSampler runtime errors:
  - "Trajectory without checkin neighbors after filtering by max_time": indicates too strict filtering or corrupted time columns.
  - "Query node index is not in graph": mismatch between indices in preprocessed tensors; verify label encoding and offsets.
- multiple_run.py uses Windows-style `set CUDA_VISIBLE_DEVICES=... && ...`.
  - On Linux, prefer: `CUDA_VISIBLE_DEVICES=0 python run.py -f best_conf/nyc.yml --multi_run_mode`


## Scripts reference

- run.py
  - Full train/validate/test with Accelerate and TensorBoard
- run_test.py
  - Utility runner with visualization helpers
- run_cold_start.py
  - Creates active/normal/inactive user splits and evaluates checkpoints; writes CSV results
- compute_acc.py
  - Scans recent log folders and aggregates reported metrics
- multiple_run.py
  - Simple loop over N runs of run.py; adapt env var syntax for your OS


## Extending the project

- New dataset
  - Add raw files under data/<new>/raw with required column names; update preprocess functions if layout differs
  - Provide/compute preprocessed artifacts and bert_address_embedding.pt
  - Add a config YAML under conf/best_conf
- New model or encoder
  - Add to model/* or layer/* and wire into model selection in run.py
  - Add new config flags under conf/README.md and consume via utils/conf_util.Cfg


## Repro tips

- Pin seeds via run_args.seed (set a specific int; multi_run_mode overrides)
- Ensure driver/toolkit matches CUDA wheels (for cu121 wheels, driver must support CUDA 12.1). CPU install is supported via requirements options swap noted above.


## Quick commands (copy/paste)

```bash
# Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train/Eval
python run.py -f best_conf/nyc.yml
# or
python run.py -f best_conf/tky.yml
python run.py -f best_conf/ca.yml

# Evaluate an existing checkpoint
python run_test.py -f best_conf/nyc.yml

# TensorBoard
tensorboard --logdir tensorboard
```