import folium
from folium.features import DivIcon
import os
import pandas as pd

def generate_poi_demo_states(row, save_dir="visulization/poi_demo_states"):
    """
    Generate 4 states of HTML maps for a trajectory:
    A: only history trajectory
    B: history trajectory + last point with question mark
    C: B + top-5 candidate points
    D: C + ground truth highlighted
    """
    os.makedirs(save_dir, exist_ok=True)

    traj_lats = row["traj_lat_list"]
    traj_lons = row["traj_lon_list"]
    top5_lats = row["top5_lat_list"]
    top5_lons = row["top5_lon_list"]
    gt_lat = row["gt_lat"]
    gt_lon = row["gt_lon"]

    traj_id = row.get("trajectory_id", "unknown")

    # basic check
    if not isinstance(traj_lats, list) or not isinstance(traj_lons, list):
        print(f"[WARN] Trajectory {traj_id} lat/lon is not list, skip.")
        return
    if len(traj_lats) < 2 or len(traj_lats) != len(traj_lons):
        print(f"[WARN] Trajectory {traj_id} lat/lon length mismatch, skip.")
        return

    # all trajectory points
    traj_points = list(zip(traj_lats, traj_lons))

    # history trajectory: remove the last point, to avoid "looking like a closed loop"
    if len(traj_points) >= 2:
        history_points = traj_points[:-1]
        last_obs_lat, last_obs_lon = history_points[-1]
    else:
        history_points = traj_points
        last_obs_lat, last_obs_lon = traj_points[-1]

    # map center
    center_lat = sum(traj_lats) / len(traj_lats)
    center_lon = sum(traj_lons) / len(traj_lons)

    # a small tool: draw the base history trajectory + p1/p2 labels
    def _base_map():
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

        # draw the history trajectory line
        folium.PolyLine(
            locations=history_points,
            weight=4,
            opacity=0.8,
            color="blue",
            tooltip=f"Trajectory {traj_id} (history)"
        ).add_to(m)

        # draw the history points + p1/p2/... labels
        for i, (lat, lon) in enumerate(history_points, start=1):
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color="blue",
                fill=True,
                fill_opacity=0.7,
                popup=f"p{i}",
                tooltip=f"p{i}",
            ).add_to(m)

            folium.Marker(
                location=[lat, lon],
                icon=DivIcon(
                    icon_size=(20, 20),
                    icon_anchor=(0, 0),
                    html=f'<div style="font-size:10px; color:blue;">p{i}</div>',
                ),
            ).add_to(m)

        return m

    # ========== state A: only draw the history trajectory ==========
    mA = _base_map()
    save_path_A = os.path.join(save_dir, f"traj_{traj_id}_A_history.html")
    mA.save(save_path_A)
    print(f"[A] Saved: {save_path_A}")

    # ========== state B: history trajectory + "?" ==========
    mB = _base_map()

    folium.Marker(
        location=[last_obs_lat, last_obs_lon],
        icon=DivIcon(
            icon_size=(20, 20),
            icon_anchor=(0, 0),
            html='<div style="font-size:18px; color:purple;">?</div>',
        ),
        popup="Where will the user go next?",
        tooltip="Next location?"
    ).add_to(mB)

    save_path_B = os.path.join(save_dir, f"traj_{traj_id}_B_question.html")
    mB.save(save_path_B)
    print(f"[B] Saved: {save_path_B}")

    # ========== state C: on the basis of B, add top-5 ==========
    mC = _base_map()

    # question mark
    folium.Marker(
        location=[last_obs_lat, last_obs_lon],
        icon=DivIcon(
            icon_size=(20, 20),
            icon_anchor=(0, 0),
            html='<div style="font-size:18px; color:purple;">?</div>',
        ),
        popup="Where will the user go next?",
        tooltip="Next location?"
    ).add_to(mC)

    # top-5 candidate points
    if isinstance(top5_lats, list) and isinstance(top5_lons, list):
        for lat, lon in zip(top5_lats, top5_lons):
            if pd.notnull(lat) and pd.notnull(lon):
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    color="green",
                    fill=True,
                    fill_opacity=0.8,
                    popup="Top-5 candidate",
                    tooltip="Top-5 candidate"
                ).add_to(mC)

    save_path_C = os.path.join(save_dir, f"traj_{traj_id}_C_top5.html")
    mC.save(save_path_C)
    print(f"[C] Saved: {save_path_C}")

    # ========== state D: on the basis of C, highlight ground truth ==========
    mD = _base_map()

    # question mark
    folium.Marker(
        location=[last_obs_lat, last_obs_lon],
        icon=DivIcon(
            icon_size=(20, 20),
            icon_anchor=(0, 0),
            html='<div style="font-size:18px; color:purple;">?</div>',
        ),
        popup="Where will the user go next?",
        tooltip="Next location?"
    ).add_to(mD)

    # top-5 candidate points
    if isinstance(top5_lats, list) and isinstance(top5_lons, list):
        for lat, lon in zip(top5_lats, top5_lons):
            if pd.notnull(lat) and pd.notnull(lon):
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    color="green",
                    fill=True,
                    fill_opacity=0.8,
                    popup="Top-5 candidate",
                    tooltip="Top-5 candidate"
                ).add_to(mD)

    # ground truth (red star + line)
    if pd.notnull(gt_lat) and pd.notnull(gt_lon):
        folium.Marker(
            location=[gt_lat, gt_lon],
            popup="Ground truth (actual next location)",
            tooltip="Ground truth",
            icon=folium.Icon(color="red", icon="star")
        ).add_to(mD)

        folium.PolyLine(
            locations=[
                [last_obs_lat, last_obs_lon],
                [gt_lat, gt_lon]
            ],
            weight=4,
            color="red",
            opacity=0.8,
            tooltip="Actual next move"
        ).add_to(mD)

    save_path_D = os.path.join(save_dir, f"traj_{traj_id}_D_gt.html")
    mD.save(save_path_D)
    print(f"[D] Saved: {save_path_D}")
