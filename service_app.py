# service_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------
# Court geometry & helper functions
# ---------------------------------------------------------------------

winner_color = {'ALCARAZ': 'royalblue', 'SINNER': 'tomato'}
side_labels = {'Right': "Deuce", 'Left': "Ad"}

# Points at the end of each set (using global point index)
annotate_ixs = [74, 154, 214, 289]

# Reference court coordinates from params.py (in centimeters)
ref_points = {
    0: (0, 0),
    1: (0, 2377),
    2: (1097, 2377),
    3: (1097, 0),
    4: (137, 0),
    5: (137, 2377),
    6: (960, 2377),
    7: (960, 0),
    8: (137, 548),
    9: (137, 1829),
    10: (960, 1829),
    11: (960, 548),
    12: (548.5, 548),
    13: (548.5, 1829)
}

def draw_full_tennis_court(ax=None):
    # [redacted, unchanged]
    if ax is None:
        ax = plt.gca()

    to_m = lambda pt: (pt[0] / 100, pt[1] / 100)

    # Court outline (outer rectangle)
    outer = [to_m(ref_points[i]) for i in [0, 1, 2, 3, 0]]
    ax.plot(*zip(*outer), color='k', lw=2)

    # Singles sidelines (4-5 and 6-7)
    ax.plot([ref_points[4][0] / 100, ref_points[5][0] / 100],
            [ref_points[4][1] / 100, ref_points[5][1] / 100], color='k', lw=2)
    ax.plot([ref_points[6][0] / 100, ref_points[7][0] / 100],
            [ref_points[6][1] / 100, ref_points[7][1] / 100], color='k', lw=2)

    # Baselines (0-3 and 1-2)
    ax.plot([ref_points[0][0] / 100, ref_points[3][0] / 100],
            [ref_points[0][1] / 100, ref_points[3][1] / 100], color='k', lw=2)
    ax.plot([ref_points[1][0] / 100, ref_points[2][0] / 100],
            [ref_points[1][1] / 100, ref_points[2][1] / 100], color='k', lw=2)

    # Service boxes - Deuce and Ad, both sides
    deuce_near = [12, 11, 10, 13]
    ad_near = [12, 8, 9, 13]

    for box in [deuce_near, ad_near]:
        poly_pts = [to_m(ref_points[i]) for i in box + [box[0]]]
        ax.plot(*zip(*poly_pts), color='k', lw=1.3)

    # Symmetric boxes on the far side (flip y)
    y_flip = lambda y: (2377 - y)
    for box in [deuce_near, ad_near]:
        pts = [(pt[0], y_flip(pt[1])) for pt in [ref_points[i] for i in box]]
        poly_pts = [to_m(p) for p in pts + [pts[0]]]
        ax.plot(*zip(*poly_pts), color='k', lw=1.3)

    # Center service line (near side)
    ax.plot([ref_points[12][0] / 100, ref_points[13][0] / 100],
            [ref_points[12][1] / 100, ref_points[13][1] / 100],
            color='k', lw=1, linestyle="--")

    # Center service line (far side, symmetry)
    ax.plot([ref_points[12][0] / 100, ref_points[13][0] / 100],
            [y_flip(ref_points[12][1]) / 100, y_flip(ref_points[13][1]) / 100],
            color='k', lw=1, linestyle="--")

    # Net
    net_y = 2377 / 2 / 100
    ax.plot([0, 10.97], [net_y, net_y], color="k", lw=1.5, linestyle="--")

    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 10.97 + 0.5)
    ax.set_ylim(-0.5, 23.77 + 0.5)  # y downwards
    ax.set_xticks([])
    ax.set_yticks([])

def serve_location_to_refpoint_coords(laterality, depth, court_side):
    # [redacted, unchanged]
    if court_side == 'Right':  # Deuce
        x0 = ref_points[12][0]
        x1 = ref_points[8][0]
        y0 = ref_points[1][1] / 2
        y1 = ref_points[13][1]
    else:  # 'Left' = Ad
        x0 = ref_points[12][0]
        x1 = ref_points[11][0]
        y0 = ref_points[1][1] / 2
        y1 = ref_points[13][1]

    # Clamp 0–100
    capped_laterality = min(max(laterality, 0), 100)
    capped_depth = min(max(depth, 0), 100)

    x = x0 + (x1 - x0) * (capped_laterality / 100.0)
    y = y0 + (y1 - y0) * (capped_depth / 100.0)
    return x / 100.0, y / 100.0

def point_to_set_number(ix: int) -> int:
    # [redacted, unchanged]
    if ix <= annotate_ixs[0]:
        return 1
    elif ix <= annotate_ixs[1]:
        return 2
    elif ix <= annotate_ixs[2]:
        return 3
    elif ix <= annotate_ixs[3]:
        return 4
    else:
        return 5

# ---------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------

st.set_page_config(page_title="Tennis Serve Explorer", layout="wide")
st.title("🎾 Tennis Serve Explorer")

st.markdown(
    "Explore serve locations, winners, tension, momentum, and sets — "
    "then build your own plots on the analysis page."
)

# ---------------- Data loading ----------------

st.sidebar.header("Data")

DATA_PATH = "match_table (1).csv"
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(f"Could not find `{DATA_PATH}` in the current directory. "
             "Make sure the file exists next to `service_app.py`.")
    st.stop()

st.sidebar.success(f"Loaded `{DATA_PATH}`")

# Basic required columns
required_cols = [
    "server_name",
    "Service",
    "serve_side",
    "tension_score",
    "rolling_momentum_alcaraz",
    "point_winner",
    "num_shot",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns in data: {missing}")
    st.stop()

# ---- Add point_index & set_number ----
point_col = None
for cand in ["ix", "point_ix", "point_index"]:
    if cand in df.columns:
        point_col = cand
        break

if point_col is not None:
    df["point_index"] = df[point_col]
else:
    df["point_index"] = df.index

df["set_number"] = df["point_index"].map(point_to_set_number)

# ---------------- Sidebar filters ----------------

st.sidebar.header("Filters")

players = sorted(df["server_name"].dropna().unique().tolist())
player = st.sidebar.selectbox("Server", players)

serve_types = sorted(df["Service"].dropna().unique().tolist())
serve_type = st.sidebar.selectbox("Serve type", serve_types)

side_choice = st.sidebar.radio(
    "Serve side",
    ["Both sides", "Deuce only", "Ad only"]
)

# Set filter (multi-select)
available_sets = sorted(df["set_number"].dropna().unique().tolist())
selected_sets = st.sidebar.multiselect(
    "Select set(s)",
    options=available_sets,
    default=available_sets
)

# Tension range
tension_min_all, tension_max_all = int(df["tension_score"].min()), int(df["tension_score"].max())
tension_min, tension_max = st.sidebar.slider("Tension score range",
                                             tension_min_all, tension_max_all,
                                             (tension_min_all, tension_max_all))

momentum_filter = st.sidebar.radio(
    "Momentum filter",
    ["All", "Alcaraz momentum", "Sinner momentum"],
)

winner_filter = st.sidebar.radio(
    "Point winner", ["All", "ALCARAZ only", "SINNER only"]
)

# Service speed filter
speed_col = f"{player}__service_speed"
if speed_col in df.columns:
    nonzero_speed = df[speed_col].replace(0, np.nan).dropna()
    if not nonzero_speed.empty:
        s_min, s_max = int(nonzero_speed.min()), int(nonzero_speed.max())
        start_min = 120
        start_min = max(s_min, start_min)
        speed_range = st.sidebar.slider(
            "Service speed (km/h)",
            s_min, s_max,
            (start_min, s_max)
        )
    else:
        speed_range = None
        st.sidebar.info("No non-zero service speed data.")
else:
    speed_range = None
    st.sidebar.info(f"No column `{speed_col}` in data.")

# ---- LATERALITY DOUBLE SLIDER FILTER ----
lat_col = f"{player}__service_laterality"
if lat_col in df.columns:
    nonzero_lats = df[lat_col].replace(0, np.nan).dropna()
    if not nonzero_lats.empty:
        lat_min, lat_max = int(nonzero_lats.min()), int(nonzero_lats.max())
        lat_filter_min = max(lat_min, 0)
        lat_filter_max = min(lat_max, 100)
        laterality_range = st.sidebar.slider(
            "Serve laterality (0=down the T, 100=wide)",
            lat_filter_min, lat_filter_max,
            (lat_filter_min, lat_filter_max)
        )
    else:
        laterality_range = None
        st.sidebar.info("No serve laterality data for this player.")
else:
    laterality_range = None
    st.sidebar.info(f"No column `{lat_col}` in data.")

# num_shot filter
ns_min_all, ns_max_all = int(df["num_shot"].min()), int(df["num_shot"].max())
finish_max = min(ns_max_all, 15)
num_shot_min, num_shot_max = st.sidebar.slider(
    "Rally length (num_shot)",
    ns_min_all, finish_max,
    (ns_min_all, finish_max)
)

# Set default to 'Scatter + Heatmap'
plot_mode = st.sidebar.radio(
    "Plot type", ["Scatter only", "Heatmap only", "Scatter + Heatmap"], index=2
)

# Fixed minimum serves for KDE
MIN_KDE_POINTS = 5

# ---------------- Apply filters (shared for both tabs) ----------------

mask = (
    (df["server_name"] == player)
    & (df["Service"] == serve_type)
    & (df["tension_score"].between(tension_min, tension_max))
    & (df["set_number"].isin(selected_sets))
    & (df["num_shot"].between(num_shot_min, num_shot_max))
)

if momentum_filter == "Alcaraz momentum":
    mask &= df["rolling_momentum_alcaraz"] > 0
elif momentum_filter == "Sinner momentum":
    mask &= df["rolling_momentum_alcaraz"] < 0

if winner_filter == "ALCARAZ only":
    mask &= df["point_winner"] == "ALCARAZ"
elif winner_filter == "SINNER only":
    mask &= df["point_winner"] == "SINNER"

if speed_range and speed_col in df.columns:
    mask &= df[speed_col].between(speed_range[0], speed_range[1])

# Apply serve laterality filter
if laterality_range and lat_col in df.columns:
    mask &= df[lat_col].between(laterality_range[0], laterality_range[1])

df_filtered = df[mask].copy()

# Map UI side choice to underlying serve_side values
if side_choice == "Deuce only":
    sides_to_plot = ["Right"]
    df_filtered = df_filtered[df_filtered["serve_side"] == "Right"]
elif side_choice == "Ad only":
    sides_to_plot = ["Left"]
    df_filtered = df_filtered[df_filtered["serve_side"] == "Left"]
else:
    sides_to_plot = ["Right", "Left"]

if df_filtered.empty:
    st.warning("No serves match the current filters.")
    st.stop()

# ---------------------------------------------------------------------
# Tabs: 1) Serve map, 2) Custom analysis
# ---------------------------------------------------------------------

tab1, tab2 = st.tabs(["🎯 Serve map & stats", "📈 Custom analysis"])

# ---------------- TAB 1: Serve map & stats ----------------
with tab1:
    st.subheader("📊 Serve statistics (on filtered points)")

    total_points = len(df_filtered)

    # Dynamically choose player for win% reporting
    win_player = player if player in ["ALCARAZ", "SINNER"] else "ALCARAZ"

    win_mask = df_filtered["point_winner"] == win_player
    win_all_pct = 100 * win_mask.mean()

    # By side (Deuce / Ad)
    df_deuce = df_filtered[df_filtered["serve_side"] == "Right"]
    df_ad = df_filtered[df_filtered["serve_side"] == "Left"]

    win_deuce_pct = 100 * (df_deuce["point_winner"] == win_player).mean() if not df_deuce.empty else None
    win_ad_pct = 100 * (df_ad["point_winner"] == win_player).mean() if not df_ad.empty else None

    # Average service speed
    avg_speed = None
    if speed_col in df_filtered.columns:
        nonzero_speed_f = df_filtered[speed_col].replace(0, np.nan).dropna()
        if not nonzero_speed_f.empty:
            avg_speed = nonzero_speed_f.mean()

    # Serve direction stats
    lat_col = f"{player}__service_laterality"
    depth_col = f"{player}__service_depth"

    t_pct = b_pct = w_pct = None
    t_win_pct = b_win_pct = w_win_pct = None

    if lat_col in df_filtered.columns and depth_col in df_filtered.columns:
        usable = df_filtered[[lat_col, depth_col, "serve_side", "point_winner"]].dropna()
        usable = usable[(usable[lat_col] != 0) & (usable[depth_col] >= 15)]
        if not usable.empty:
            lats = usable[lat_col].values

            is_t = lats <= 33
            is_b = (lats > 33) & (lats <= 66)
            is_w = lats > 66

            t_pct = 100 * np.mean(is_t)
            b_pct = 100 * np.mean(is_b)
            w_pct = 100 * np.mean(is_w)

            t_points = usable[is_t]
            b_points = usable[is_b]
            w_points = usable[is_w]

            t_win_pct = 100 * (t_points["point_winner"] == win_player).mean() if len(t_points) > 0 else None
            b_win_pct = 100 * (b_points["point_winner"] == win_player).mean() if len(b_points) > 0 else None
            w_win_pct = 100 * (w_points["point_winner"] == win_player).mean() if len(w_points) > 0 else None

    col1, col2 = st.columns(2)

    with col1:
        speed_value = f"{avg_speed:.1f} km/h" if avg_speed is not None else "N/A"

        if win_player == "SINNER":
            all_label = "Sinner win % (all)"
            deuce_label = "Sinner win % (Deuce serves)"
            ad_label = "Sinner win % (Ad serves)"
        else:
            all_label = "Alcaraz win % (all)"
            deuce_label = "Alcaraz win % (Deuce serves)"
            ad_label = "Alcaraz win % (Ad serves)"

        stats_data = [
            ["Total filtered points", total_points, ""],
            ["Avg service speed", speed_value, ""],
            [all_label, f"{win_all_pct:.1f}%", ""],
        ]
        if win_deuce_pct is not None:
            stats_data.append([deuce_label, f"{win_deuce_pct:.1f}%", ""])
        if win_ad_pct is not None:
            stats_data.append([ad_label, f"{win_ad_pct:.1f}%", ""])

        st.table(
            {
                "Stat": [row[0] for row in stats_data],
                "Value": [row[1] for row in stats_data],
                "": [row[2] for row in stats_data],
            }
        )

    with col2:
        serve_dir_rows = []
        if t_pct is not None:
            t_pct_text = f"{t_pct:.1f}%"
            b_pct_text = f"{b_pct:.1f}%"
            w_pct_text = f"{w_pct:.1f}%"

            t_win_text = f"{t_win_pct:.1f}%" if t_win_pct is not None else "N/A"
            b_win_text = f"{b_win_pct:.1f}%" if b_win_pct is not None else "N/A"
            w_win_text = f"{w_win_pct:.1f}%" if w_win_pct is not None else "N/A"

            serve_dir_rows = [
                ["T", t_pct_text, t_win_text],
                ["Body", b_pct_text, b_win_text],
                ["Wide", w_pct_text, w_win_text],
            ]
        else:
            serve_dir_rows = [["Serve direction", "N/A", "N/A"]]

        win_label = "% Sinner wins" if win_player == "SINNER" else "% Alcaraz wins"

        st.table(
            {
                "Serve direction (T / Body / Wide)": [row[0] for row in serve_dir_rows],
                "Proportion": [row[1] for row in serve_dir_rows],
                win_label: [row[2] for row in serve_dir_rows],
            }
        )

    # ---- Serve map plot ----
    fig, ax = plt.subplots(figsize=(4, 8))
    draw_full_tennis_court(ax)

    ax.set_title(
        f"{player} — {serve_type}\n"
        f"Serves (tension {tension_min}–{tension_max}, sets {selected_sets})",
        fontsize=9.8,
    )

    for court_side in sides_to_plot:
        side_df = df_filtered[df_filtered["serve_side"] == court_side].copy()
        if side_df.empty:
            continue

        x_key = f"{player}__service_laterality"
        y_key = f"{player}__service_depth"
        if x_key not in side_df.columns or y_key not in side_df.columns:
            continue

        coords = side_df[[x_key, y_key]].dropna()
        coords = coords[(coords[x_key] != 0) & (coords[y_key] >= 15)]
        if coords.empty:
            continue

        xs, ys, valid_idx = [], [], []
        for idx, row in coords.iterrows():
            x, y = serve_location_to_refpoint_coords(row[x_key], row[y_key], court_side)
            xs.append(x)
            ys.append(y)
            valid_idx.append(idx)

        if not xs:
            continue

        sub = side_df.loc[valid_idx]
        colors = sub["point_winner"].map(winner_color).fillna("gray").tolist()

        if plot_mode in ("Scatter only", "Scatter + Heatmap"):
            ax.scatter(
                xs,
                ys,
                c=colors,
                alpha=0.7,
                edgecolors="k",
                s=25,
                label=f"{side_labels.get(court_side, court_side)}",
                zorder=5,
            )

        if plot_mode in ("Heatmap only", "Scatter + Heatmap") and len(xs) >= MIN_KDE_POINTS:
            try:
                sns.kdeplot(
                    x=xs,
                    y=ys,
                    fill=True,
                    cmap="Reds" if court_side == "Right" else "Blues",
                    alpha=0.25,
                    bw_adjust=0.7,
                    thresh=0.01,
                    ax=ax,
                    zorder=2,
                )
            except Exception:
                pass

    al_patch = mpatches.Patch(color=winner_color["ALCARAZ"], label="ALCARAZ wins point")
    si_patch = mpatches.Patch(color=winner_color["SINNER"], label="SINNER wins point")
    ax.legend(handles=[al_patch, si_patch], loc="upper right", framealpha=0.9, fontsize=7)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
    plt.tight_layout()

    st.pyplot(fig)


# ---------------- TAB 2: Custom analysis ----------------
with tab2:
    st.subheader("📈 Custom scatter + quadratic regression")

    # Options for y and x axes
    y_candidates = {
        "Serve speed": f"{player}__service_speed",
        "Serve depth": f"{player}__service_depth",
        "Serve laterality": f"{player}__service_laterality",
        "Alcaraz % of points won (binned)": "__alcaraz_win_pct__",  # Special handler
    }

    y_options = []
    for label, col in y_candidates.items():
        if col == "__alcaraz_win_pct__":
            # Always allow this derived axis if point_winner exists and x axis is a number
            if "point_winner" in df_filtered.columns:
                y_options.append(label)
        else:
            if col in df_filtered.columns:
                y_options.append(label)

    # Add serve speed to the x_candidates for selection
    x_candidates = {
        "Tension score": "tension_score",
        "Point index": "point_index",
    }
    speed_x_col = f"{player}__service_speed"
    if speed_x_col in df_filtered.columns:
        x_candidates["Serve speed"] = speed_x_col

    x_options = [label for label, col in x_candidates.items() if col in df_filtered.columns]

    if not y_options or not x_options:
        st.warning("Cannot build custom plot: missing required columns in filtered data.")
    else:
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            y_label = st.selectbox("Y axis", y_options)
        with col_sel2:
            x_label = st.selectbox("X axis", x_options)

        x_col = x_candidates[x_label]
        y_col = y_candidates[y_label]

        # ------ Start Custom filter: Remove 2nd serves over 200 km/h ------
        # Only for the custom analysis tab!
        serve_speed_col = f"{player}__service_speed"
        # Service order is coded as `Service == '2nd Serve'`
        if serve_speed_col in df_filtered.columns and "Service" in df_filtered.columns:
            df_filtered_custom = df_filtered[
                ~(
                    (df_filtered["Service"] == "2nd Serve")
                    & (df_filtered[serve_speed_col] > 200)
                )
            ].copy()
        else:
            df_filtered_custom = df_filtered.copy()
        # ------ End custom filter ------

        # --- Enable fit by point winner: Single fit or two fits
        st.markdown("**Quadratic fit options:**")
        fit_mode = st.radio(
            "Quadratic fit mode",
            ["One fit (all points)", "Two fits (by point winner)"],
            index=0,
            help="If 'Two fits', fits a separate quadratic for each value of point_winner."
        )

        if y_col == "__alcaraz_win_pct__":
            # Derive percentage of points won by Alcaraz, binned by x_col
            plot_df = df_filtered_custom[[x_col, "point_winner"]].replace([np.inf, -np.inf], np.nan)
            plot_df = plot_df.dropna(subset=[x_col, "point_winner"])
            if plot_df.empty:
                st.warning("No data left after filtering for the selected X axis.")
            else:
                # --- Custom bins for Serve Speed ----
                use_speed_bins = (x_col == f"{player}__service_speed")
                bin_centers = []
                alcaraz_win_pcts = []
                bin_sizes = []
                if use_speed_bins:
                    speeds = plot_df[x_col].values.astype(float)
                    min_bin_start = 150
                    bin_width = 30
                    bin_step = 10
                    max_speed = np.ceil(speeds.max()) if len(speeds) > 0 else min_bin_start+bin_width
                    bin_edges = []
                    current = min_bin_start
                    while (current + bin_width) < max_speed+bin_width:
                        bin_edges.append((current, current+bin_width))
                        current += bin_step
                    for l, h in bin_edges:
                        group = plot_df[(plot_df[x_col] >= l) & (plot_df[x_col] < h)]
                        bin_center = (l + h) / 2
                        bin_centers.append(bin_center)
                        bin_sizes.append(len(group))
                        if len(group) > 0:
                            win_pct = np.mean(group["point_winner"] == "ALCARAZ") * 100
                        else:
                            win_pct = np.nan
                        alcaraz_win_pcts.append(win_pct)
                else:
                    n_bins = 8
                    uniq_vals = plot_df[x_col].nunique()
                    if uniq_vals < n_bins:
                        bins = sorted(plot_df[x_col].unique())
                        grouped = plot_df.groupby(x_col)
                        for val in bins:
                            group = grouped.get_group(val)
                            bin_centers.append(val)
                            win_pct = np.mean(group["point_winner"] == "ALCARAZ") * 100
                            alcaraz_win_pcts.append(win_pct)
                            bin_sizes.append(len(group))
                    else:
                        bin_counts, bin_edges = np.histogram(plot_df[x_col], bins=n_bins)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        alcaraz_win_pcts = []
                        bin_sizes = []
                        inds = np.digitize(plot_df[x_col], bin_edges) - 1
                        for i in range(len(bin_centers)):
                            group = plot_df[(inds == i)]
                            bin_sizes.append(len(group))
                            if len(group) > 0:
                                win_pct = np.mean(group["point_winner"] == "ALCARAZ") * 100
                            else:
                                win_pct = np.nan
                            alcaraz_win_pcts.append(win_pct)

                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.plot(bin_centers, alcaraz_win_pcts, marker="o", color="royalblue", label="Alcaraz % points won")
                ax2.set_ylabel("Alcaraz % of points won")
                ax2.set_xlabel(x_label)
                ax2.set_title(f"Alcaraz % of points won vs {x_label}\n(filtered with the same settings as the serve map)")
                ax2.grid(True, alpha=0.3)
                for xc, ypct, size in zip(bin_centers, alcaraz_win_pcts, bin_sizes):
                    if not np.isnan(ypct):
                        ax2.annotate(f"{int(size)}", (xc, ypct), textcoords="offset points", xytext=(0,5), ha='center', fontsize=7, color='gray')
                bin_centers_valid = np.array([c for c, v in zip(bin_centers, alcaraz_win_pcts) if not np.isnan(v)])
                win_pcts_valid = np.array([v for v in alcaraz_win_pcts if not np.isnan(v)])
                # Single fit only (does not make sense to "split fits" for an aggregated percentage line)
                if len(bin_centers_valid) >= 3:
                    try:
                        coeffs = np.polyfit(bin_centers_valid, win_pcts_valid, 2)
                        x_line = np.linspace(np.min(bin_centers_valid), np.max(bin_centers_valid), 200)
                        y_line = np.polyval(coeffs, x_line)
                        ax2.plot(x_line, y_line, linewidth=2, label="Quadratic fit", color='orange')
                    except Exception:
                        pass
                ax2.legend()
                plt.tight_layout()
                st.pyplot(fig2)
        else:
            # Standard single-point mode as before
            plot_df = df_filtered_custom[[x_col, y_col, "point_winner"]].replace([np.inf, -np.inf], np.nan)
            plot_df = plot_df.dropna(subset=[x_col, y_col])

            if plot_df.empty:
                st.warning("No data left after filtering for the selected X/Y axes.")
            else:
                colors = plot_df["point_winner"].map(winner_color).fillna("gray")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.scatter(
                    plot_df[x_col],
                    plot_df[y_col],
                    c=colors,
                    alpha=0.6,
                    edgecolors="k",
                    s=25,
                    label=None,
                )

                if len(plot_df) >= 3:
                    if fit_mode == "One fit (all points)":
                        x_vals = plot_df[x_col].values.astype(float)
                        y_vals = plot_df[y_col].values.astype(float)
                        try:
                            coeffs = np.polyfit(x_vals, y_vals, 2)
                            x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
                            y_line = np.polyval(coeffs, x_line)
                            ax2.plot(x_line, y_line, linewidth=2, label="Quadratic fit")
                        except Exception:
                            pass
                    elif fit_mode == "Two fits (by point winner)":
                        winners = ["ALCARAZ", "SINNER"]
                        fit_colors = {"ALCARAZ": "royalblue", "SINNER": "tomato"}
                        # Just show "ALCARAZ" and "SINNER" colors, no "Quadratic (ALCARAZ wins)" labels
                        for winner in winners:
                            sub = plot_df[plot_df["point_winner"] == winner]
                            if len(sub) >= 3:
                                x_sub = sub[x_col].values.astype(float)
                                y_sub = sub[y_col].values.astype(float)
                                try:
                                    coeffs = np.polyfit(x_sub, y_sub, 2)
                                    x_line = np.linspace(x_sub.min(), x_sub.max(), 200)
                                    y_line = np.polyval(coeffs, x_line)
                                    l = ax2.plot(
                                        x_line, y_line, linewidth=2, color=fit_colors[winner],
                                        # No "Quadratic ..." label here
                                    )
                                    # Only legend for point-winner color, not quadratic fits:
                                except Exception:
                                    continue
                        # Add main point-winner color legend too
                        al_patch = mpatches.Patch(color=winner_color["ALCARAZ"], label="ALCARAZ wins point")
                        si_patch = mpatches.Patch(color=winner_color["SINNER"], label="SINNER wins point")
                        legend_handles = [al_patch, si_patch]
                        ax2.legend(handles=legend_handles, loc="best", framealpha=0.9)
                    else:
                        pass  # fallback, should not be needed
                if fit_mode == "One fit (all points)":
                    # Only show color legend for point winner
                    al_patch = mpatches.Patch(color=winner_color["ALCARAZ"], label="ALCARAZ wins point")
                    si_patch = mpatches.Patch(color=winner_color["SINNER"], label="SINNER wins point")
                    ax2.legend(handles=[al_patch, si_patch], loc="best", framealpha=0.9)

                ax2.set_xlabel(x_label)
                ax2.set_ylabel(y_label)
                ax2.set_title(f"{y_label} vs {x_label}\n(filtered with the same settings as the serve map)")
                ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)
