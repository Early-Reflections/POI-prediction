# POI Prediction System - Complete Architecture and Data Flow

Codemap ID: POI_Prediction_System_-_Complete_Architecture_and_Data_Flow_20251111_132053

Description: Complete documentation of the Context-aware Spatiotemporal Graph Attention Network for POI recommendation, covering data preprocessing [2a-2e], model architecture [3a-3e], attention mechanisms [4a-4e], hypergraph construction [5a-5e], and evaluation pipeline [6a-6e].


## Trace 1: Training Pipeline Execution

Text diagram:
```
Training Pipeline Execution
├── Configuration System <-- conf_util.py:15
│   └── Parse YAML config file <-- 1a
├── Data Processing Pipeline <-- run.py:18
│   ├── Execute preprocessing on raw data <-- 1b
│   └── Load preprocessed hypergraph data <-- 1c
├── Model System <-- run.py:157
│   ├── Create STHGCN model instance <-- 1d
│   └── Training Loop <-- run.py:221
│       ├── Batch sampling <-- run.py:89
│       ├── Forward pass execution <-- 1e
│       ├── Loss computation <-- run.py:241
│       └── Backward propagation <-- run.py:244
└── Evaluation System <-- run.py:248
    ├── Validation during training <-- run.py:262
    └── Final test evaluation <-- run.py:315
```

Location map:
- 1a: Configuration Loading — /home/one/Documents/gitkraken/POI-prediction/run.py:26
- 1b: Data Preprocessing Trigger — /home/one/Documents/gitkraken/POI-prediction/run.py:67
- 1c: Dataset Initialization — /home/one/Documents/gitkraken/POI-prediction/run.py:71
- 1d: Model Instantiation — /home/one/Documents/gitkraken/POI-prediction/run.py:158
- 1e: Forward Pass Execution — /home/one/Documents/gitkraken/POI-prediction/run.py:240


## Trace 2: Data Preprocessing and Hypergraph Construction

Text diagram:
```
Data Preprocessing Pipeline
├── preprocess() main entry <-- preprocess_main.py:10
│   ├── FileReader.read_dataset() <-- 2a
│   │   └── Parse raw LBSN data <-- file_reader.py:36
│   ├── FileReader.do_filter() <-- 2b
│   │   └── Filter low-frequency entities <-- file_reader.py:56
│   ├── FileReader.generate_id() <-- 2c
│   │   ├── Create pseudo-session trajectories <-- file_reader.py:131
│   │   └── Label encoding for entities <-- file_reader.py:142
│   └── generate_hypergraph_from_file() <-- 2d
│       ├── generate_hyperedge_stat() <-- generate_hypergraph.py:103
│       │   └── Compute trajectory statistics <-- generate_hypergraph.py:103
│       ├── generate_ci2traj_pyg_data() <-- 2e
│       │   └── Build checkin-trajectory graph <-- generate_hypergraph.py:167
│       ├── generate_traj2traj_data() <-- generate_hypergraph.py:196
│       │   ├── intra-user relationships <-- generate_hypergraph.py:65
│       │   └── inter-user relationships <-- generate_hypergraph.py:75
│       └── merge_traj2traj_data() <-- generate_hypergraph.py:313
│           └── Combine trajectory graphs <-- generate_hypergraph.py:347
└── Output: PyG hypergraph data files <-- generate_hypergraph.py:84
```

Location map:
- 2a: Raw Data Loading — /home/one/Documents/gitkraken/POI-prediction/preprocess/preprocess_main.py:15
- 2b: Frequency Filtering — /home/one/Documents/gitkraken/POI-prediction/preprocess/preprocess_main.py:18
- 2c: Trajectory Generation — /home/one/Documents/gitkraken/POI-prediction/preprocess/preprocess_main.py:21
- 2d: Hypergraph Construction — /home/one/Documents/gitkraken/POI-prediction/preprocess/preprocess_main.py:27
- 2e: Checkin-to-Trajectory Graph — /home/one/Documents/gitkraken/POI-prediction/preprocess/generate_hypergraph.py:57


## Trace 3: Model Forward Pass Architecture

Text diagram:
```
STHGCN Model Forward Pass <-- sthgcn.py:183
├── Input Data Processing
│   ├── Extract check-in features <-- sthgcn.py:188
│   │   └── Feature embedding generation <-- 3a
│   ├── Temporal encoding
│   │   └── Time difference encoding <-- 3b
│   └── Spatial encoding
│       └── Distance encoding <-- 3c
├── Hypergraph Convolution <-- sthgcn.py:216
│   ├── Time-filtered convolution
│   │   └── Apply conv_for_time_filter <-- 3d
│   └── Trajectory-to-trajectory conv <-- sthgcn.py:230
│       └── Multi-hop message passing <-- sthgcn.py:232
└── Output Generation <-- sthgcn.py:282
    └── Final POI prediction layer <-- 3e
```

Location map:
- 3a: Feature Embedding — /home/one/Documents/gitkraken/POI-prediction/model/sthgcn.py:189
- 3b: Temporal Encoding — /home/one/Documents/gitkraken/POI-prediction/model/sthgcn.py:201
- 3c: Spatial Encoding — /home/one/Documents/gitkraken/POI-prediction/model/sthgcn.py:207
- 3d: Time-Filtered Convolution — /home/one/Documents/gitkraken/POI-prediction/model/sthgcn.py:217
- 3e: POI Prediction — /home/one/Documents/gitkraken/POI-prediction/model/sthgcn.py:283


## Trace 4: Hypergraph Transformer Attention Mechanism

Text diagram:
```
HypergraphTransformer Message Passing
├── forward() main entry point <-- conv.py:217
│   ├── propagate() message passing init <-- conv.py:241
│   │   └── message() computation <-- 4a
│   │       ├── Low-rank key projection
│   │       │   └── torch.matmul(x_j, U_key) <-- 4a
│   │       ├── Key transformation
│   │       │   └── torch.matmul(low_rank_key, V_key) <-- 4b
│   │       ├── Query computation (if needed) <-- conv.py:450
│   │       └── Attention score calculation
│   │           ├── Scaled dot-product <-- 4c
│   │           │   └── (query * key).sum() / sqrt() <-- 4c
│   │           └── Softmax normalization <-- 4d
│   │               └── softmax(alpha, index, ptr, size_i) <-- 4d
│   └── Output processing <-- conv.py:262
│       ├── Value projection & weighting <-- conv.py:461
│       │   └── out * alpha.unsqueeze(-1) <-- 4e
│       └── Return final attention output <-- conv.py:306
└── Feature attention integration <-- conv.py:435
    └── feature_attention() preprocessing <-- conv.py:436
```

Location map:
- 4a: Low-Rank Key Projection — /home/one/Documents/gitkraken/POI-prediction/layer/conv.py:439
- 4b: Key Transformation — /home/one/Documents/gitkraken/POI-prediction/layer/conv.py:440
- 4c: Attention Score Computation — /home/one/Documents/gitkraken/POI-prediction/layer/conv.py:454
- 4d: Attention Normalization — /home/one/Documents/gitkraken/POI-prediction/layer/conv.py:457
- 4e: Attention-Weighted Output — /home/one/Documents/gitkraken/POI-prediction/layer/conv.py:466


## Trace 5: Trajectory-to-Trajectory Relationship Construction

Text diagram:
```
Trajectory Relationship Construction Pipeline
├── generate_traj2traj_data() main function <-- generate_hypergraph.py:196
│   ├── Create trajectory-POI sparse matrix <-- 5a
│   │   └── Compute trajectory similarities <-- 5b
│   ├── Apply temporal ordering constraints <-- 5c
│   ├── Calculate spatial distances between
│   │   └── trajectory centroids using haversine <-- 5d
│   └── merge_traj2traj_data() final assembly <-- generate_hypergraph.py:313
│       ├── Build trajectory features <-- generate_hypergraph.py:338
│       └── Create edge attributes with size &
│           └── similarity metrics <-- 5e
└── filter_chunk() similarity thresholding <-- generate_hypergraph.py:394
    ├── Process trajectory pairs in chunks <-- generate_hypergraph.py:424
    └── Apply Jaccard/min-size filtering <-- generate_hypergraph.py:430
```

Location map:
- 5a: Trajectory-POI Matrix — /home/one/Documents/gitkraken/POI-prediction/preprocess/generate_hypergraph.py:225
- 5b: Trajectory Similarity Computation — /home/one/Documents/gitkraken/POI-prediction/preprocess/generate_hypergraph.py:240
- 5c: Temporal Constraint Filtering — /home/one/Documents/gitkraken/POI-prediction/preprocess/generate_hypergraph.py:261
- 5d: Spatial Distance Calculation — /home/one/Documents/gitkraken/POI-prediction/preprocess/generate_hypergraph.py:303
- 5e: Edge Attribute Construction — /home/one/Documents/gitkraken/POI-prediction/preprocess/generate_hypergraph.py:381


## Trace 6: Model Evaluation and Metrics

Text diagram:
```
Model Evaluation Pipeline <-- run.py:315
├── Load trained checkpoint <-- 6a
│   └── Restore model state <-- 6b
├── Execute test step function <-- run.py:320
│   ├── Model forward pass <-- pipeline_util.py:109
│   │   └── Generate predictions <-- pipeline_util.py:109
│   ├── Sort predictions descending <-- 6d
│   └── Calculate ranking metrics <-- 6c
│       ├── Recall@K computation <-- 6e
│       ├── NDCG@K computation <-- pipeline_util.py:120
│       ├── MAP@K computation <-- pipeline_util.py:121
│       └── MRR computation <-- pipeline_util.py:123
└── Log evaluation results <-- run.py:338
```

Location map:
- 6a: Model Loading — /home/one/Documents/gitkraken/POI-prediction/run.py:318
- 6b: State Restoration — /home/one/Documents/gitkraken/POI-prediction/run.py:319
- 6c: Test Execution — /home/one/Documents/gitkraken/POI-prediction/run.py:320
- 6d: Ranking Generation — /home/one/Documents/gitkraken/POI-prediction/utils/pipeline_util.py:111
- 6e: Recall Computation — /home/one/Documents/gitkraken/POI-prediction/utils/pipeline_util.py:119


## Code snippet anchors (from codemap)

- run.py
  - Lines 24-28: parse YAML and sizes list
  - Lines 65-73: preprocess + dataset init
  - Lines 156-160: model selection
  - Lines 238-242: forward + loss + backward start
  - Lines 316-322: load checkpoint and evaluate

- preprocess/preprocess_main.py
  - Lines 13-23: imports and distance helper
  - Lines 25-29: NYC raw file loading

- preprocess/generate_hypergraph.py
  - Lines 55-59: generate stats and ci2traj
  - Lines 223-227: build traj-POI matrix
  - Lines 238-242: compute traj2traj
  - Lines 259-263: temporal filtering
  - Lines 301-305: haversine and logging
  - Lines 379-383: edge_attr assembly

- model/sthgcn.py
  - Lines 187-191: extract features and zeros for traj
  - Lines 199-203: time encoder for ci2traj
  - Lines 205-209: distance encoder
  - Lines 215-219: time-filtered conv
  - Lines 281-285: logits and loss

- layer/conv.py
  - Lines 437-442: low-rank key projection
  - Lines 452-459: query-key attention and softmax
  - Lines 464-468: weighted values output

- utils/pipeline_util.py
  - Lines 109-113: evaluation forward + ranking
  - Lines 117-121: metrics computation
