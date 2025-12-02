import json
import os
import os.path as osp
from pathlib import Path
from typing import Dict, List

import pandas as pd


DATASET_NAME = "nyc"
TOP_K = 5
MIN_TRAJ_LEN = 4
MAX_TRAJ_LEN = 20
MAX_PER_CLASS = 15  # cap per class; actual count may be lower


def _find_latest_predictions(root_dir: str, dataset: str) -> str:
    log_dir = osp.join(root_dir, "log")
    if not osp.isdir(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    candidates: List[str] = []
    for ts in os.listdir(log_dir):
        d = osp.join(log_dir, ts, dataset)
        fp = osp.join(d, "test_predictions_top20.csv")
        if osp.isfile(fp):
            candidates.append(fp)
    if not candidates:
        raise FileNotFoundError("No test_predictions_top20.csv found under log/*/%s" % dataset)
    candidates.sort(key=lambda p: osp.getmtime(p), reverse=True)
    return candidates[0]


def _load_data(root_dir: str, dataset: str, predictions_path: str | None = None):
    if predictions_path is None:
        predictions_path = _find_latest_predictions(root_dir, dataset)
    print(f"[export_nyc_demo] Using predictions file: {predictions_path}")

    pred_df = pd.read_csv(predictions_path)

    sample_path = osp.join(root_dir, "data", dataset, "preprocessed", "sample.csv")
    sample_df = pd.read_csv(
        sample_path,
        usecols=[
            "trajectory_id",
            "UTCTimeOffsetEpoch",
            "Latitude",
            "Longitude",
        ],
    )

    poi_db_path = osp.join(root_dir, "data", dataset, "preprocessed", "POI_database.csv")
    poi_df = pd.read_csv(poi_db_path)
    poi_coord: Dict[int, tuple[float, float]] = {}
    for r in poi_df.itertuples(index=False):
        try:
            pid = int(getattr(r, "poi_id"))
            lat = float(getattr(r, "latitude"))
            lon = float(getattr(r, "longitude"))
        except Exception:
            continue
        poi_coord[pid] = (lat, lon)

    return pred_df, sample_df, poi_coord


def _classify(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_cols = [f"pred_top_{i}" for i in range(1, TOP_K + 1)]
    pred_df = pred_df.copy()
    pred_df["top1_hit"] = pred_df["PoiId"] == pred_df["pred_top_1"]
    hit_any = pred_df[pred_cols].eq(pred_df["PoiId"], axis=0).any(axis=1)
    pred_df["topk_hit"] = hit_any

    pred_df["hit_class"] = "miss"
    pred_df.loc[pred_df["topk_hit"], "hit_class"] = "top5"
    pred_df.loc[pred_df["top1_hit"], "hit_class"] = "top1"
    return pred_df


def _filter_region(pred_df: pd.DataFrame) -> pd.DataFrame:
    # Keep trajectories roughly around Manhattan / nearby for nicer camera motion.
    def in_box(lat: float, lon: float) -> bool:
        return 40.60 <= lat <= 40.82 and -74.20 <= lon <= -73.90

    m = pred_df.apply(lambda r: in_box(r["Latitude"], r["Longitude"]), axis=1)
    return pred_df[m].copy()


def build_scenarios(pred_df: pd.DataFrame, sample_df: pd.DataFrame, poi_coord: Dict[int, tuple[float, float]]):
    # Classify hits on the full prediction set for global metrics.
    pred_all = _classify(pred_df)

    stats_overall = {
        "top1_rate": float(pred_all["top1_hit"].mean()) if len(pred_all) else 0.0,
        "top5_rate": float(pred_all["topk_hit"].mean()) if len(pred_all) else 0.0,
        "total_samples": int(len(pred_all)),
    }

    # For now, use the full NYC prediction set as the pool for scenario selection.
    # If we want to constrain to a smaller map area, we can re-enable _filter_region here.
    pred_pool = pred_all.reset_index(drop=True)

    # Shuffle once with a fixed seed for reproducible scenario selection
    shuffled = pred_pool.sample(frac=1.0, random_state=42).reset_index(drop=True)

    groups = {}
    for cls in ("top1", "top5", "miss"):
        groups[cls] = shuffled[shuffled["hit_class"] == cls]

    scenarios = []
    counts_by_class = {"top1": 0, "top5": 0, "miss": 0}

    # Pre-index sample by trajectory_id for speed
    grouped_sample = sample_df.groupby("trajectory_id")

    for cls in ("top1", "top5", "miss"):
        df_cls = groups[cls]
        if df_cls.empty:
            continue
        for _, row in df_cls.iterrows():
            if counts_by_class[cls] >= MAX_PER_CLASS:
                break
            traj_id = int(row["trajectory_id"])
            cutoff = float(row["UTCTimeOffsetEpoch"])
            if traj_id not in grouped_sample.groups:
                continue
            hist = grouped_sample.get_group(traj_id)
            hist = hist[hist["UTCTimeOffsetEpoch"] <= cutoff].sort_values("UTCTimeOffsetEpoch")
            if hist.empty:
                continue
            if not (MIN_TRAJ_LEN <= len(hist) <= MAX_TRAJ_LEN):
                continue

            points = [
                {"lat": float(lat), "lon": float(lon)}
                for lat, lon in zip(hist["Latitude"], hist["Longitude"])
            ]

            # Map top-K predictions to coordinates
            predicted_pois = []
            for rank in range(1, TOP_K + 1):
                col = f"pred_top_{rank}"
                if col not in row:
                    continue
                try:
                    poi_id = int(row[col])
                except Exception:
                    continue
                coord = poi_coord.get(poi_id)
                if coord is None:
                    continue
                score = (TOP_K + 1 - rank) / TOP_K  # simple monotonically decreasing proxy
                predicted_pois.append(
                    {
                        "lat": float(coord[0]),
                        "lon": float(coord[1]),
                        "score": float(score),
                        "rank": int(rank),
                    }
                )

            if not predicted_pois:
                continue

            gt_poi_id = int(row["PoiId"])
            scenario_id = f"nyc-{traj_id}-{cls}"
            name_suffix = {
                "top1": "top-1 hit",
                "top5": "top-5 hit",
                "miss": "miss",
            }[cls]
            scenario = {
                "id": scenario_id,
                "name": f"NYC trajectory {traj_id} ({name_suffix})",
                "trajectory_id": traj_id,
                "user_id": int(row["UserId"]),
                "hit_class": cls,
                "ground_truth_poi": gt_poi_id,
                "points": points,
                "predictedPois": predicted_pois,
            }
            scenarios.append(scenario)
            counts_by_class[cls] += 1

    return scenarios, stats_overall, counts_by_class


def main(dataset: str, predictions_path: str | None = None) -> None:
    root = str(Path(__file__).resolve().parents[1])
    pred_df, sample_df, poi_coord = _load_data(root, dataset, predictions_path)
    scenarios, stats_overall, counts_by_class = build_scenarios(pred_df, sample_df, poi_coord)

    out_path = osp.join(root, "mapping", "web", "public", "data", f"demo-{dataset}-real.json")
    payload = {
        "dataset": dataset,
        "source": "test_predictions_top20.csv",
        "stats_overall": stats_overall,
        "scenario_counts": counts_by_class,
        "scenarios": scenarios,
    }
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[export_nyc_demo] Wrote {len(scenarios)} scenarios to {out_path}")
    print(f"[export_nyc_demo] Overall top-1 rate in filtered pool: {stats_overall['top1_rate']:.3f}")
    print(f"[export_nyc_demo] Overall top-{TOP_K} rate in filtered pool: {stats_overall['top5_rate']:.3f}")
    print(f"[export_nyc_demo] Counts by class: {counts_by_class}")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Export real NYC demo trajectories + predictions for mapping UI.")
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Optional explicit path to test_predictions_top20.csv; defaults to newest under log/*/nyc.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nyc",
    )
    args = parser.parse_args()
    main(args.dataset, args.predictions)
