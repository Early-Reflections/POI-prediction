import os
import matplotlib.pyplot as plt
import numpy as np
import folium
from folium.features import DivIcon
import os
import pandas as pd
from utils import get_root_dir
from gene_video import generate_poi_demo_states # 导入生成视频的函数

# ===== drop by matplotlib, without map visualization=====
def plot_trajectory_with_predictions(row, save_dir="visulization/poi_demo_plots"):
    os.makedirs(save_dir, exist_ok=True)

    traj_lats = row["traj_lat_list"]
    traj_lons = row["traj_lon_list"]
    top5_lats = row["top5_lat_list"]
    top5_lons = row["top5_lon_list"]
    gt_lat = row["gt_lat"]
    gt_lon = row["gt_lon"]

    traj_id = row.get("trajectory_id", "unknown")

    plt.figure(figsize=(6, 6))

    # 1) draw the user trajectory (line + points)
    if len(traj_lats) > 1 and len(traj_lats) == len(traj_lons):
        # trajectory line
        plt.plot(traj_lons, traj_lats, marker="o", linewidth=1.5, alpha=0.8)
        # start point and end point
        plt.scatter(traj_lons[0], traj_lats[0], s=60, marker="s", label="Start")
        plt.scatter(traj_lons[-2], traj_lats[-2], s=60, marker="o", label="Last observed")
    else:
        print(f"Trajectory {traj_id} has invalid lat/lon lists.")

    # 2) draw the top-5 candidate points
    if len(top5_lats) == len(top5_lons) and len(top5_lats) > 0:
        plt.scatter(top5_lons, top5_lats, s=50, alpha=0.8, label="Top-5 candidates")

    # 3) draw the ground truth (highlight)
    if pd.notnull(gt_lat) and pd.notnull(gt_lon):
        plt.scatter(gt_lon, gt_lat, s=80, marker="*", label="Ground truth", edgecolors="black", linewidths=1.5)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Trajectory {traj_id} with Top-5 Predictions")

    plt.legend(loc="best")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"traj_{traj_id}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")

# ===== drop by folium, with map visualization=====
def plot_trajectory_with_predictions_folium(row, save_dir="visulization/poi_demo_maps"):
    os.makedirs(save_dir, exist_ok=True)

    traj_lats = row["traj_lat_list"]
    traj_lons = row["traj_lon_list"]
    top5_lats = row["top5_lat_list"]
    top5_lons = row["top5_lon_list"]
    gt_lat = row["gt_lat"]
    gt_lon = row["gt_lon"]

    traj_id = row.get("trajectory_id", "unknown")

    # check the trajectory validity
    if not isinstance(traj_lats, list) or not isinstance(traj_lons, list):
        print(f"Trajectory {traj_id} has invalid lat/lon lists.")
        return
    if len(traj_lats) < 2 or len(traj_lats) != len(traj_lons):
        print(f"Trajectory {traj_id} has invalid lat/lon length.")
        return

    # 1) map center = trajectory center
    center_lat = sum(traj_lats) / len(traj_lats)
    center_lon = sum(traj_lons) / len(traj_lons)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # 2) draw the entire trajectory line
    traj_points = list(zip(traj_lats, traj_lons))  # (lat, lon)
    folium.PolyLine(
        locations=traj_points,
        weight=4,
        opacity=0.8,
        tooltip=f"Trajectory {traj_id}"
    ).add_to(m)

    # 3) draw each point on the history trajectory, and mark p1, p2, p3... in order
    for i, (lat, lon) in enumerate(traj_points, start=1):
        # small circle (optional)
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="blue",
            fill=True,
            fill_opacity=0.7,
            popup=f"p{i}",
            tooltip=f"p{i}",
        ).add_to(m)

        # text label: display "p1" "p2" directly on the map
        folium.Marker(
            location=[lat, lon],
            icon=DivIcon(
                icon_size=(20, 20),
                icon_anchor=(0, 0),
                html=f'<div style="font-size:10px; color:blue;">p{i}</div>',
            ),
        ).add_to(m)

    # 4) mark the start point and the last observed point (can reuse p1/pN, or add separately)
    start_lat, start_lon = traj_points[0]
    last_lat, last_lon = traj_points[-1]

    folium.Marker(
        location=[start_lat, start_lon],
        popup="Start",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)

    folium.Marker(
        location=[last_lat, last_lon],
        popup="Last observed",
        icon=folium.Icon(color="orange", icon="pause")
    ).add_to(m)

    # 5) top-5 candidate points (green)
    if isinstance(top5_lats, list) and isinstance(top5_lons, list):
        for lat, lon in zip(top5_lats, top5_lons):
            if pd.notnull(lat) and pd.notnull(lon):
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    color="green",
                    fill=True,
                    fill_opacity=0.8,
                    popup="Top-5 candidate"
                ).add_to(m)

    # 6) ground truth (red star)
    if pd.notnull(gt_lat) and pd.notnull(gt_lon):
        folium.Marker(
            location=[gt_lat, gt_lon],
            popup="Ground truth",
            icon=folium.Icon(color="red", icon="star")
        ).add_to(m)

    # 7) save html
    save_path = os.path.join(save_dir, f"traj_{traj_id}.html")
    m.save(save_path)
    print(f"Saved folium map: {save_path}")

if __name__ == '__main__':

    # read the test data with the top 20 predictions,
    # this file is generated by the run_test.py script
    df = pd.read_csv('change_to_your_path/test_predictions_top20.csv')

    target_cols = ['pred_top_1', 'pred_top_2', 'pred_top_3', 'pred_top_4', 'pred_top_5']

    df_hit = df[df.apply(lambda row: row['PoiId'] in row[target_cols].tolist(), axis=1)]

    print(f"Hit count: {len(df_hit)} of {len(df)}")

    # read the data\nyc\preprocessed\sample.csv
    # change the path to your own path
    df_sample = pd.read_csv('change_to_your_path/sample.csv')

    # then select the rows where the trajectory_id is in the trajectory_id of df_sample
    df_sample_hit = df_sample[df_sample['trajectory_id'].isin(df_hit['trajectory_id'])]
    df_points = df_sample_hit.sort_values(["trajectory_id", "UTCTime"])
    # group by trajectory_id and aggregate the latitude and longitude into a list
    df_traj = df_points.groupby("trajectory_id").agg(
        traj_lat_list=("Latitude", list),
        traj_lon_list=("Longitude", list),
        gt_lat=("Latitude", lambda x: x.iloc[-1]),
        gt_lon=("Longitude", lambda x: x.iloc[-1]),
    ).reset_index()

    # df_traj merge df_hit
    df_traj = pd.merge(df_traj, df_hit, on='trajectory_id')

    print(df_traj.head(0))

    # get the latitude and longitude of the poi_id in the 'pred_top_1', 'pred_top_2', 'pred_top_3', 'pred_top_4', 'pred_top_5' columns of df_hit, based on the PoiId in the sample.csv
    poi_coord = (
        df_sample[['PoiId', 'Latitude', 'Longitude']]
        .drop_duplicates(subset='PoiId')
        .set_index('PoiId')
    )

    def get_top5_coord_lists(row):
        lat_list = []
        lon_list = []
        for col in target_cols:
            poi_id = row[col]
            if poi_id in poi_coord.index:
                lat_list.append(poi_coord.loc[poi_id, 'Latitude'])
                lon_list.append(poi_coord.loc[poi_id, 'Longitude'])
            else:
                lat_list.append(np.nan)
                lon_list.append(np.nan)
        return pd.Series({'top5_lat_list': lat_list, 'top5_lon_list': lon_list})

    df_hit_with_top5 = df_traj.copy()
    df_hit_with_top5[['top5_lat_list', 'top5_lon_list']] = df_hit_with_top5.apply(
        get_top5_coord_lists,
        axis=1
    )

    print(df_hit_with_top5.head(0))

    # 1) add a trajectory length column
    df_hit_with_top5["traj_len"] = df_hit_with_top5["traj_lat_list"].apply(len)

    # 2) define a function to check if the last N points have duplicates
    def tail_has_duplicates(row, tail=4, tol=1e-5):
        lats = row["traj_lat_list"]
        lons = row["traj_lon_list"]
        if len(lats) < tail:
            # too short trajectory is considered as "no duplicates", can be filtered later by length
            return False

        tail_lats = lats[-tail:]
        tail_lons = lons[-tail:]

        # round the coordinates to avoid floating point differences
        coords = [(round(lat, 5), round(lon, 5)) for lat, lon in zip(tail_lats, tail_lons)]
        return len(coords) != len(set(coords))  # if there are duplicates, return True

    # 3) filter: length between [4, 20], and the last few steps have no duplicates
    df_clean = df_hit_with_top5[
        df_hit_with_top5["traj_len"].between(4, 20) &
        (~df_hit_with_top5.apply(tail_has_duplicates, axis=1))
    ].copy()

    print(f"After filtering: {len(df_clean)} trajectories left (from {len(df_hit_with_top5)})")


    # ===== random sample 10 trajectories to plot =====
    df_sample_for_plot = df_clean.sample(10, random_state=42)

    for _, row in df_sample_for_plot.iterrows():
        plot_trajectory_with_predictions(row)

    # ===== random sample 10 trajectories to plot folium =====
    df_sample_for_plot = df_clean.sample(10, random_state=42)  # 或 df_hit_with_top5
    for _, row in df_sample_for_plot.iterrows():
        plot_trajectory_with_predictions_folium(row)

    # select the trajectory_id in df_clean == 4649 to generate the video states

    df_4649 = df_clean[df_clean['trajectory_id'] == 4649]
    row = df_4649.iloc[0]
    # generate the video states
    generate_poi_demo_states(row)