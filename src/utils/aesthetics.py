import numpy as np
import re
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

def color_mapping(design, metric_lbl=None):
    """
    Map mult-arm randomization design names to colors
    """
    # Benchmarks are blue scheme
    if re.match(pattern=r"^(?:CR|CR (Sch))" , string=design) is not None:
        return "navy"
    if re.match(pattern=r"CR (Set)" , string=design) is not None:
        return "grey"
    if re.match(pattern=r"^(?:QB|GR|GFR)" , string=design) is not None:
        return "cornflowerblue"

    if metric_lbl is None:
        match_str = design
    else:
        match_str = f"{design} - {metric_lbl}"
    # IGR with balance metrics are orange scheme
    if re.match(pattern=r"^(?:IGR|IGR (GFR)) - MaxMahalanobis$", string=match_str) is not None:
        return "lightsalmon"
    if re.match(pattern=r"^(?:IGR|IGR (GFR)) - SumMaxAbsSMD$", string=match_str) is not None:
        return "goldenrod"
    if re.match(pattern=r"^(?:IGRg|IGRg (GFR)) - MaxMahalanobis$", string=match_str) is not None:
        return "orangered"
    if re.match(pattern=r"^(?:IGRg|IGRg (GFR)) - SumMaxAbsSMD$", string=match_str) is not None:
        return "sienna"

    # IGR with aggregated fitness fn, including an exposure metric, are green scheme 
    if re.match(pattern=r"^IGR - (.*?)FracExpo$", string=match_str) is not None:
        return "darkseagreen"
    if re.match(pattern=r"^IGRg - (.*?)FracExpo$", string=match_str) is not None:
        return "seagreen"
    
    # IGR with aggregated fitness fn, including a distance metric, are purple scheme
    if re.match(pattern=r"^IGR - (.*?)InvMinEuclidDist$", string=match_str) is not None:
        # return "thistle"
        return "orchid"
    if re.match(pattern=r"^IGRg - (.*?)InvMinEuclidDist$", string=match_str) is not None:
        # return "mediumorchid"
        return "purple"

def two_tone_color_mapping(design): 
    if r"$\mathcal{Z}_{pool}$" in design:
        return "silver"
    elif r"$\mathcal{Z}^{*}_{pool}$" in design:
        return "dimgrey"

def get_palette(designs, metric_lbl=None):
    """
    Create palette dictionary mapping randomization design names to colors
    """
    if isinstance(designs, str):
        return {designs: color_mapping(designs, metric_lbl)}
    else:
        return {
            design: color_mapping(design, metric_lbl)
            for design in designs
        }

def get_two_tone_palette(enum_types):
    return {
        enum_type: two_tone_color_mapping(enum_type)
        for enum_type in enum_types
    }

def get_hue_order(rand_mdl_names):
    complete_names = [
        name for name in rand_mdl_names if "CR" in name and "IGR" not in name
    ]
    block_names = [
        name for name in rand_mdl_names if "QB" in name and "IGR" not in name
    ]
    graph_names = [
        name for name in rand_mdl_names if "GR" in name and "IGR" not in name
    ]
    comp_names = [
        name for name in rand_mdl_names if "GFR" in name and "IGR" not in name
    ]

    restricted_name_pairs = []
    rand_mdl_names = sorted(rand_mdl_names)
    for name in rand_mdl_names:
        if "IGR" in name and "IGRg" not in name:
            igrg_name = name.replace("IGR", "IGRg")
            if igrg_name in rand_mdl_names:
                restricted_name_pairs.extend([name, igrg_name])
            else:
                restricted_name_pairs.append(name)

    ordered_names = (
        complete_names
        + block_names
        + graph_names
        + comp_names
        + restricted_name_pairs
    )

    return ordered_names

def setup_fig(ncols, sharex, sharey, cbar=False):
    fig, axs = plt.subplots( 
        1,  ncols, figsize=(8 * ncols, 8),
        sharex=sharex, sharey=sharey,
    )
    if cbar:
        plt.subplots_adjust(wspace=0.5)
    if ncols == 1:
        axs = [axs]
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=16)

    return fig, axs

def format_ax(ax, lbl_size=16):
    max_x = ax.get_xlim()[1]
    max_y = ax.get_ylim()[1]

    if max_x > 100:
        x_fmt = "{x:,.1e}"
    elif max_x > 10:
        x_fmt = "{x:,.1f}"
    elif max_x < 0.1:
        x_fmt = "{x:,.3f}"
    else:
        x_fmt = "{x:,.2f}" 

    if max_y > 100:
        y_fmt = "{x:,.1e}"
    elif max_y > 10:
        y_fmt = "{x:,.1f}"
    elif max_y < 0.1:
        y_fmt = "{x:,.3f}"
    else:
        y_fmt = "{x:,.2f}"

    ax.xaxis.set_major_formatter(StrMethodFormatter(x_fmt))
    ax.yaxis.set_major_formatter(StrMethodFormatter(y_fmt))
    ax.tick_params(axis='both', which='major', labelsize=lbl_size)

    return ax

def adjust_joint_grid_limits(jnt_grids, save_dir, save_fnames):
    xmin = np.min([jnt_grid.ax_joint.get_xlim()[0] for jnt_grid in jnt_grids])
    xmax = np.max([jnt_grid.ax_joint.get_xlim()[1] for jnt_grid in jnt_grids])
    y_max = np.max([jnt_grid.ax_joint.get_ylim()[1] for jnt_grid in jnt_grids])
    y_min = np.min([jnt_grid.ax_joint.get_ylim()[0] for jnt_grid in jnt_grids])
    for jnt_grid, save_fname in zip(jnt_grids, save_fnames):
        jnt_grid.ax_joint.set_xlim(xmin, xmax)
        jnt_grid.ax_joint.set_ylim(y_min, y_max)
        jnt_grid.savefig(save_dir / save_fname, transparent=True, bbox_inches="tight")
