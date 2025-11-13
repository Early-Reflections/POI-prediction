import argparse
import os
import os.path as osp
import pickle
import time
from typing import Optional, List

import numpy as np
import pandas as pd
import torch

from preprocess import preprocess
from utils import Cfg, get_root_dir, seed_torch
from dataset import LBSNDataset
from layer import NeighborSampler
from model import STHGCN, SequentialTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Predict top-K next POIs for a single user")
    parser.add_argument("--user_id", type=int, required=True, help="Target user id (encoded in dataset)")
    parser.add_argument("-f", "--yaml_file", type=str, default="best_conf/nyc.yml", help="Config file relative to conf/")
    parser.add_argument("-k", "--top_k", type=int, default=5, help="Top-K predictions to return")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory containing checkpoint.pt")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index override; if not set, uses config's run_args.gpu")
    parser.add_argument("--all", action="store_true", help="Predict for all test samples of the user (default: only most recent)")
    parser.add_argument("--names", action="store_true", help="Also print category names inferred from raw data")
    parser.add_argument("--gps", action="store_true", help="Also print GPS coordinates (lat, lon) inferred from raw data")
    return parser.parse_args()


def find_latest_checkpoint(root_dir: str, dataset_name: str) -> Optional[str]:
    tb_dir = osp.join(root_dir, "tensorboard")
    if not osp.isdir(tb_dir):
        return None
    candidates: List[str] = []
    for ts in os.listdir(tb_dir):
        base = osp.join(tb_dir, ts, dataset_name)
        if osp.isdir(base) and osp.isfile(osp.join(base, "checkpoint.pt")):
            candidates.append(base)
    if not candidates:
        return None
    candidates.sort(key=lambda p: osp.getmtime(p), reverse=True)
    return candidates[0]


def maybe_load_poi_inverse_encoder(preprocessed_path: str):
    pkl_path = osp.join(preprocessed_path, "label_encoding.pkl")
    if not osp.isfile(pkl_path):
        return None, None
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        poi_label_encoder = data[0]
        pad_id = data[5]
        classes = getattr(poi_label_encoder, 'classes_', None)
        return classes, pad_id
    except Exception:
        return None, None


def build_sampler_for_indices(cfg: Cfg, lbsn_dataset: LBSNDataset, indices: List[int]) -> NeighborSampler:
    sizes = cfg.model_args.sizes if isinstance(cfg.model_args.sizes, (list, tuple)) else [int(i) for i in cfg.model_args.sizes.split('-')]
    device = 'cuda:' + str(cfg.run_args.gpu) if int(cfg.run_args.gpu) >= 0 else 'cpu'
    sampler = NeighborSampler(
        lbsn_dataset.x,
        lbsn_dataset.edge_index,
        lbsn_dataset.edge_attr,
        intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
        inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
        edge_t=lbsn_dataset.edge_t,
        edge_delta_t=lbsn_dataset.edge_delta_t,
        edge_type=lbsn_dataset.edge_type,
        sizes=sizes,
        sample_idx=torch.tensor(indices, dtype=torch.long),  # subset of global sample indices
        node_idx=lbsn_dataset.node_idx_test,  # must be full tensor; sampler indexes by sample_idx values
        edge_delta_s=lbsn_dataset.edge_delta_s,
        max_time=lbsn_dataset.max_time_test,  # full tensor aligned with sample indices
        label=lbsn_dataset.label_test,  # full tensor aligned with sample indices
        batch_size=min(len(indices), int(cfg.run_args.eval_batch_size)),
        num_workers=0 if device == 'cpu' else int(cfg.run_args.num_workers),
        shuffle=False,
        pin_memory=True,
    )
    return sampler


def maybe_build_poi_name_lookup(root_dir: str, dataset_name: str):
    raw_dir = osp.join(root_dir, 'data', dataset_name, 'raw')
    if not osp.isdir(raw_dir):
        return None
    import glob
    paths = []
    # Prefer known filenames if present; otherwise, fall back to all csv/txt
    preferred = [
        'NYC_train.csv', 'NYC_val.csv', 'NYC_test.csv',
        'dataset_TSMC2014_TKY.txt',
        'dataset_gowalla_ca_ne.csv'
    ]
    for p in preferred:
        fp = osp.join(raw_dir, p)
        if osp.isfile(fp):
            paths.append(fp)
    if not paths:
        paths = glob.glob(osp.join(raw_dir, '*.csv')) + glob.glob(osp.join(raw_dir, '*.txt'))
    if not paths:
        return None
    name_map = {}
    for fp in paths:
        try:
            if fp.endswith('.txt'):
                df = pd.read_csv(fp, sep='\t', encoding='latin-1')
            else:
                df = pd.read_csv(fp)
        except Exception:
            continue
        cols_lower = {c.lower(): c for c in df.columns}
        # Find ID column (case-insensitive, support variants)
        id_candidates = [
            'poiid', 'poi_id', 'venueid', 'venue_id', 'id'
        ]
        id_col = None
        for k in id_candidates:
            if k in cols_lower:
                id_col = cols_lower[k]
                break
        # Legacy exact match
        if id_col is None and 'PoiId' in df.columns:
            id_col = 'PoiId'
        if id_col is None:
            continue
        # Prefer human-readable name; fall back to category string, then category id
        name_candidates = [
            'venue_name', 'venuename', 'name',
            'poicatname', 'poi_catname', 'venuecategory',
            'poicategoryname', 'poicategoryid'
        ]
        name_col = None
        for k in name_candidates:
            if k in cols_lower:
                name_col = cols_lower[k]
                break
        # Legacy exact matches
        if name_col is None and 'PoiCategoryName' in df.columns:
            name_col = 'PoiCategoryName'
        if name_col is None and 'PoiCategoryId' in df.columns:
            name_col = 'PoiCategoryId'
        if name_col is None:
            continue
        sub = df[[id_col, name_col]].dropna()
        # Do not overwrite existing mapping; first occurrence wins
        for r in sub.itertuples(index=False):
            pid = r[0]
            # Normalize to plain Python types when possible
            try:
                pid_py = pid.item()  # numpy scalar -> python
            except Exception:
                pid_py = pid
            if pid_py not in name_map:
                name_map[pid_py] = str(r[1])
    return name_map if name_map else None


def maybe_build_poi_gps_lookup(root_dir: str, dataset_name: str):
    raw_dir = osp.join(root_dir, 'data', dataset_name, 'raw')
    if not osp.isdir(raw_dir):
        return None
    import glob
    paths = []
    preferred = [
        'NYC_train.csv', 'NYC_val.csv', 'NYC_test.csv',
        'dataset_TSMC2014_TKY.txt',
        'dataset_gowalla_ca_ne.csv'
    ]
    for p in preferred:
        fp = osp.join(raw_dir, p)
        if osp.isfile(fp):
            paths.append(fp)
    if not paths:
        paths = glob.glob(osp.join(raw_dir, '*.csv')) + glob.glob(osp.join(raw_dir, '*.txt'))
    if not paths:
        return None
    gps_map = {}
    for fp in paths:
        try:
            if fp.endswith('.txt'):
                df = pd.read_csv(fp, sep='\t', encoding='latin-1')
            else:
                df = pd.read_csv(fp)
        except Exception:
            continue
        cols_lower = {c.lower(): c for c in df.columns}
        id_candidates = ['poiid', 'poi_id', 'venueid', 'venue_id', 'id']
        id_col = None
        for k in id_candidates:
            if k in cols_lower:
                id_col = cols_lower[k]
                break
        if id_col is None and 'PoiId' in df.columns:
            id_col = 'PoiId'
        if id_col is None:
            continue
        lat_candidates = ['latitude', 'lat']
        lon_candidates = ['longitude', 'lon', 'lng']
        lat_col = None
        lon_col = None
        for k in lat_candidates:
            if k in cols_lower:
                lat_col = cols_lower[k]
                break
        for k in lon_candidates:
            if k in cols_lower:
                lon_col = cols_lower[k]
                break
        # Legacy exact matches
        if lat_col is None and 'Latitude' in df.columns:
            lat_col = 'Latitude'
        if lon_col is None and 'Longitude' in df.columns:
            lon_col = 'Longitude'
        if lat_col is None or lon_col is None:
            continue
        sub = df[[id_col, lat_col, lon_col]].dropna()
        for r in sub.itertuples(index=False):
            pid = r[0]
            try:
                pid_py = pid.item()
            except Exception:
                pid_py = pid
            if pid_py not in gps_map:
                try:
                    lat_v = float(r[1])
                    lon_v = float(r[2])
                except Exception:
                    continue
                gps_map[pid_py] = (lat_v, lon_v)
    return gps_map if gps_map else None


def main():
    args = parse_args()
    root = get_root_dir()

    # Load config and set device
    cfg = Cfg(args.yaml_file)
    if args.gpu is not None:
        cfg.run_args.gpu = args.gpu
    device = 'cuda:' + str(cfg.run_args.gpu) if int(cfg.run_args.gpu) >= 0 else 'cpu'
    cfg.run_args.device = device  # required by model init
    # Ensure sizes is a list for model configuration
    if isinstance(cfg.model_args.sizes, str):
        cfg.model_args.sizes = [int(i) for i in cfg.model_args.sizes.split('-')]

    # Seed
    seed = int(time.time()) % 1000000
    seed_torch(seed)

    # Ensure data exists (preprocess is idempotent)
    preprocess(cfg)

    # Load dataset
    lbsn_dataset = LBSNDataset(cfg)
    # Populate dataset-derived fields needed by the model
    cfg.dataset_args.spatial_slots = lbsn_dataset.spatial_slots
    cfg.dataset_args.num_user = lbsn_dataset.num_user
    cfg.dataset_args.num_poi = lbsn_dataset.num_poi
    cfg.dataset_args.num_category = lbsn_dataset.num_category
    cfg.dataset_args.padding_poi_id = lbsn_dataset.padding_poi_id
    cfg.dataset_args.padding_user_id = lbsn_dataset.padding_user_id
    cfg.dataset_args.padding_poi_category = lbsn_dataset.padding_poi_category
    cfg.dataset_args.padding_hour_id = lbsn_dataset.padding_hour_id
    cfg.dataset_args.padding_weekday_id = lbsn_dataset.padding_weekday_id

    # Determine indices for the target user from test set
    pre_path = osp.join(root, 'data', cfg.dataset_args.dataset_name, 'preprocessed')
    test_csv = osp.join(pre_path, 'test_sample.csv')
    if not osp.isfile(test_csv):
        raise FileNotFoundError(f"Missing test_sample.csv at {test_csv}")
    df_test = pd.read_csv(test_csv)

    # Filter by UserId (encoded id)
    user_rows = df_test[df_test['UserId'] == args.user_id]
    if user_rows.empty:
        print(f"No test samples found for user_id={args.user_id} in dataset={cfg.dataset_args.dataset_name}.")
        return

    if args.all:
        indices = user_rows.index.tolist()
    else:
        # Default: only the most recent sample for this user (last row)
        indices = [user_rows.index.tolist()[-1]]

    # Build sampler for selected indices
    sampler = build_sampler_for_indices(cfg, lbsn_dataset, indices)

    # Build model
    if cfg.model_args.model_name == 'sthgcn':
        model = STHGCN(cfg)
    elif cfg.model_args.model_name == 'seq_transformer':
        model = SequentialTransformer(cfg)
    else:
        raise NotImplementedError(f"Unknown model: {cfg.model_args.model_name}")
    model = model.to(device)
    model.eval()

    # Resolve checkpoint
    ckpt_dir = args.checkpoint or find_latest_checkpoint(root, cfg.dataset_args.dataset_name)
    if ckpt_dir is None:
        raise FileNotFoundError("No checkpoint found. Provide --checkpoint or train a model first.")
    ckpt_path = osp.join(ckpt_dir, 'checkpoint.pt')
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    # Optional: inverse map POI ids to original ids
    poi_classes, pad_id = maybe_load_poi_inverse_encoder(pre_path)
    name_map = maybe_build_poi_name_lookup(root, cfg.dataset_args.dataset_name) if args.names else None
    gps_map = maybe_build_poi_gps_lookup(root, cfg.dataset_args.dataset_name) if args.gps else None

    print("=== Single-User Prediction ===")
    print(f"Dataset      : {cfg.dataset_args.dataset_name}")
    print(f"User (encoded): {args.user_id}")
    print(f"Config       : {args.yaml_file}")
    print(f"Checkpoint   : {ckpt_dir}")
    print(f"Top-K        : {args.top_k}")
    print(f"Samples      : {'all' if args.all else 'latest only'} ({len(indices)} sample(s))\n")

    with torch.no_grad():
        for batch in sampler:
            split_index = torch.max(batch.adjs_t[1].storage.row()).tolist()
            batch = batch.to(device)
            input_data = {
                'x': batch.x,
                'edge_index': batch.adjs_t,
                'edge_attr': batch.edge_attrs,
                'split_index': split_index,
                'delta_ts': batch.edge_delta_ts,
                'delta_ss': batch.edge_delta_ss,
                'edge_type': batch.edge_types,
            }
            logits, _ = model(input_data, label=batch.y[:, 0], mode='test')
            topk_scores, topk_idx = torch.topk(logits, k=min(args.top_k, logits.shape[1]), dim=1)

            # Labels (encoded)
            true_labels = batch.y[:, 0].long().detach().cpu().numpy()

            for i in range(topk_idx.shape[0]):
                pred_encoded = topk_idx[i].detach().cpu().numpy()
                if poi_classes is not None:
                    classes_len = len(poi_classes)
                    pred_original = [poi_classes[int(j)] if int(j) < classes_len else None for j in pred_encoded]
                    t = int(true_labels[i])
                    true_original = poi_classes[t] if t < classes_len else None
                else:
                    pred_original = None
                    true_original = None

                probs = torch.softmax(logits[i], dim=-1)[topk_idx[i]].detach().cpu().numpy().tolist()

                print(f"Sample {int(batch.sample_idx[i])}: ")
                if pred_original is not None:
                    print(f"  True POI (encoded/original): {int(true_labels[i])} / {true_original}")
                    print(f"  Top-{args.top_k} (encoded): {pred_encoded.tolist()}")
                    print(f"  Top-{args.top_k} (original ids): {pred_original}")
                    if name_map is not None:
                        names = []
                        for pid in pred_original:
                            if pid is None:
                                names.append(None)
                                continue
                            name = name_map.get(pid)
                            if name is None:
                                # Try alternate key representations
                                try:
                                    name = name_map.get(int(pid))
                                except Exception:
                                    pass
                                if name is None:
                                    name = name_map.get(str(pid))
                            names.append(name)
                        print(f"  Top-{args.top_k} (names): {names}")
                    if gps_map is not None:
                        gps_coords = []
                        for pid in pred_original:
                            if pid is None:
                                gps_coords.append(None)
                                continue
                            coord = gps_map.get(pid)
                            if coord is None:
                                try:
                                    coord = gps_map.get(int(pid))
                                except Exception:
                                    pass
                                if coord is None:
                                    coord = gps_map.get(str(pid))
                            gps_coords.append(coord)
                        print(f"  Top-{args.top_k} (gps): {gps_coords}")
                    print(f"  Top-{args.top_k} (scores): {[round(p, 4) for p in probs]}\n")
                else:
                    print(f"  True POI (encoded): {int(true_labels[i])}")
                    print(f"  Top-{args.top_k} (encoded): {pred_encoded.tolist()}")
                    print(f"  Top-{args.top_k} (scores): {[round(p, 4) for p in probs]}\n")


if __name__ == "__main__":
    main()
