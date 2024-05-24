
import numpy as np
import folium
from matplotlib import pyplot as plt
import seaborn as sns

def plot_kenya_school_locs(sch_coords, fig_dir):
    set_palette = {0: 'blue', 1: 'orange', 2: 'green', 3: 'red', 4: 'purple'}
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True)

    school_info = sch_coords.groupby('school_id')[['coord_lon', 'coord_lat', 'settlement_id']].agg(
        coord_lon = ('coord_lon', 'mean'),
        coord_lat = ('coord_lat', 'mean'),
        n = ('coord_lon', 'size'),
        settlement_id = ('settlement_id', 'first')).reset_index(drop=True)

    centroid = np.mean(school_info[['coord_lon', 'coord_lat']], axis=0)
    m = folium.Map(
        location=(centroid[0], centroid[1]),
        tiles="Cartodb Positron",
        zoom_start=12,
    )
    max_n = np.max(school_info['n'])
    for _, info in school_info.iterrows():
        coords = (info['coord_lon'], info['coord_lat'])
        n = info['n']
        folium.CircleMarker(
            location=(coords[0], coords[1]),
            radius=10 * n / max_n,
            fill=True,
            stroke=False,
            fill_opacity=0.6,
            tooltip = f"coords = ({coords[0]}, {coords[1]}), n = {n}",
            color = set_palette[info['settlement_id']]
        ).add_to(m)

    save_path = fig_dir / f"school_locs.html"
    print(save_path)
    m.save(save_path)

def plot_y0_distr_across_sets(y_0, X, fig_dir):
    X['y0'] = y_0
    # y0_means = X.groupby(['settlement_id'])['y0'].mean().reset_index()
    n_sets = len(X['settlement_id'].unique())
    bw = (y_0.max() - y_0.min()) / 30

    fig, ax = plt.subplots(1, n_sets, figsize=(10*n_sets, 10), sharex=True, sharey=True)

    for i, set_name in enumerate(X['settlement'].unique()):
        sns.histplot(
            X[X['settlement'] == set_name]['y0'],
            binwidth=bw,
            kde=True,
            stat='frequency',
            ax=ax[i]
        )
        ax[i].set_title(f"{set_name.capitalize()}", fontsize=22)
        ax[i].set_xlabel(r"$Y_0$", fontsize=22)
        ax[i].set_ylabel("Frequency", fontsize=22)
        ax[i].tick_params(axis="x", labelsize=18)
        ax[i].tick_params(axis="y", labelsize=18)

    save_path = fig_dir / f"y0_distr_across_sets.png"
    print(save_path)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return

def plot_deg_distr_across_sets(X, X_school, A, fig_dir):
    n_sets = len(X['settlement_id'].unique())
    degs = A.sum(axis=1)

    fig, ax = plt.subplots(1, n_sets, figsize=(10*n_sets, 10), sharex=True, sharey=True)
    bw = (degs.max() - degs.min()) / 30

    for i, set_name in enumerate(X['settlement'].unique()):
        sns.histplot(
            degs[X['settlement'] == set_name],
            binwidth=bw,
            kde=True,
            line_kws={"linewidth": 3, 
                      "bw_adjust": 2},
            stat='frequency',
            ax=ax[i]
        )
        set_name_cap = set_name.capitalize()
        ax[i].set_title(f"{set_name_cap}", fontsize=22)
        ax[i].set_xlabel(r"Degree", fontsize=22)
        ax[i].set_ylabel("Frequency", fontsize=22)
        ax[i].tick_params(axis="x", labelsize=18)
        ax[i].tick_params(axis="y", labelsize=18)

    save_path = fig_dir / f"deg_distr_across_sets.png"
    print(save_path)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()