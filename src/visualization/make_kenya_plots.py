import numpy as np
import argparse
from pathlib import Path
import pickle
import folium
from matplotlib import pyplot as plt
import seaborn as sns

from src.sims.run_kenya_trial import *
from src.sims.run_multarm_trial import *

from src.sims.trial import *

def _plot_kenya_adjacency(data, fig_dir):
    _, _, X, _, X_school, _, A = data
    
    n = X_school.shape[0]
    mapping = X['school_id'].values

    W = np.zeros((n, n))
    for schl in range(n):
        indivs_in_schl_mask = mapping == schl
        indivs_in_schl_adj = np.sum(A[indivs_in_schl_mask, ], axis=0)
        schl_adj = np.bincount(mapping, weights=indivs_in_schl_adj)
        W[schl, ] = schl_adj

    plt.imsave(fig_dir / f"school_adjacency.png", W)

def _plot_kenya_adj_v_dist(data, fig_dir):
    _, _, X, _, X_school, _, A = data
    n = X_school.shape[0]
    pairwise_dists = np.zeros((n, n))
    for (i, j) in combinations(range(n), 2):
        i_coords = X_school.iloc[i][["coord_lat", "coord_lon"]]
        j_coords = X_school.iloc[j][["coord_lat", "coord_lon"]]
        dist = np.linalg.norm(i_coords - j_coords)
        pairwise_dists[i, j] = dist
        pairwise_dists[j, i] = dist

    mapping = X['school_id'].values
    W = np.zeros((n, n))
    for schl in range(n):
        indivs_in_schl_mask = mapping == schl
        indivs_in_schl_adj = np.sum(A[indivs_in_schl_mask, ], axis=0)
        schl_adj = np.bincount(mapping, weights=indivs_in_schl_adj)
        W[schl, ] = schl_adj
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.scatterplot(x=pairwise_dists.flatten(), y=W.flatten(), ax=ax)
    ax.set_xlabel("Pairwise Distance", fontsize=16)
    ax.set_ylabel("Adjacency", fontsize=16)
    ax.tick_params(axis='x', labelsize=14) 
    ax.tick_params(axis='y', labelsize=14)
    fig.savefig(fig_dir / f"adj_v_dist.png", dpi=300, bbox_inches='tight')


def _plot_kenya_school_locs(data, fig_dir):
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True)

    X = data[2]

    school_info = X.groupby('school_id')[['coord_lon', 'coord_lat']].agg(
        coord_lon = ('coord_lon', 'mean'),
        coord_lat = ('coord_lat', 'mean'),
        n = ('coord_lon', 'size')).reset_index(drop=True)

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
            tooltip = f"coords = ({coords[0]}, {coords[1]}), n = {n}"
        ).add_to(m)
    m.save(fig_dir / f"school_locs.html")

def _plot_kenya_covariate_distr_across_sets(data, fig_dir):
    X = data[2]
    vars = X.columns[X.columns.str.startswith('x')]
    fig, ax = plt.subplots(len(vars), 1, figsize=(10, 10*len(vars)))

    for i, var in enumerate(vars):
        sns.boxplot(data = X, x="settlement", y=var, ax=ax[i])

    fig.savefig(fig_dir / f"covariate_distr.png", dpi=300, bbox_inches='tight')

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--exp-subdir', type=str, default='kenya')
    parser.add_argument('--param-subdir', type=str, default='params_sigma_scale-0.01')
    parser.add_argument('--outcome-subdir', type=str, default='outcome-kenya-hierarchical_tau-0.30')

    parser.add_argument('--fig-types', type=str, nargs='+')
    parser.add_argument('--fig-dir', type=str, default='figs')
    
    args = parser.parse_args()

    data_dir = Path(args.data_dir) / args.exp_subdir / args.param_subdir / args.outcome_subdir

    for net_mdl_subdir in data_dir.glob('net-*'):
        for data_fname in net_mdl_subdir.glob(f'*.pkl'):
            with open(data_fname, 'rb') as f:
                data = pickle.load(f)
            save_dir = Path(args.fig_dir) / args.exp_subdir / args.param_subdir / args.outcome_subdir / net_mdl_subdir.stem / data_fname.stem
            if not save_dir.exists():
                save_dir.mkdir(parents=True)

            if 'school_locs' in args.fig_types:
                _plot_kenya_school_locs(data, save_dir)
            if 'covariate_distr_across_sets' in args.fig_types:
                _plot_kenya_covariate_distr_across_sets(data, save_dir)
            if 'adjacency' in  args.fig_types:
                _plot_kenya_adjacency(data, save_dir)
            if 'adj_v_dist' in args.fig_types:
                _plot_kenya_adj_v_dist(data, save_dir)



    