import numpy as np
import re
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from pathlib import Path
import seaborn as sns
from typing import List


def design_color_mapping(design: str, fitness_lbl: str = None) -> str:
    """
    Map randomization design names to colors
    Args:
        - design: Randomization design name
        - fitness_lbl: Name of fitness function used in randomization design
    """
    # Benchmarks are blue scheme
    if re.match(pattern=r"^(?:CR|CR (Sch))", string=design) is not None:
        return "navy"
    if re.match(pattern=r"CR (Set)", string=design) is not None:
        return "grey"
    if re.match(pattern=r"^(?:QB|GR|GFR)", string=design) is not None:
        return "cornflowerblue"

    if fitness_lbl is None:
        match_str = design
    else:
        match_str = f"{design} - {fitness_lbl}"
    # IGR with balance metrics are orange scheme
    if (
        re.match(pattern=r"^(?:IGR|IGR (GFR)) - MaxMahalanobis$", string=match_str)
        is not None
    ):
        return "lightsalmon"
    if (
        re.match(pattern=r"^(?:IGR|IGR (GFR)) - SumMaxAbsSMD$", string=match_str)
        is not None
    ):
        return "goldenrod"
    if (
        re.match(pattern=r"^(?:IGRg|IGRg (GFR)) - MaxMahalanobis$", string=match_str)
        is not None
    ):
        return "orangered"
    if (
        re.match(pattern=r"^(?:IGRg|IGRg (GFR)) - SumMaxAbsSMD$", string=match_str)
        is not None
    ):
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
    if (
        re.match(pattern=r"^IGRg - (.*?)InvMinEuclidDist$", string=match_str)
        is not None
    ):
        # return "mediumorchid"
        return "purple"

    else:
        raise ValueError(f"Unknown design and metric: {design}, {fitness_lbl}")


def get_design_palette(designs: List[str], fitness_lbl: str = None) -> dict:
    """
    Create palette dictionary mapping designs to colors
    Args:
        - designs: List of randomization designs
        - fitness_lbl: Name of fitness function used in randomization design
    """
    if isinstance(designs, str):
        return {designs: design_color_mapping(designs, fitness_lbl)}
    else:
        return {design: design_color_mapping(design, fitness_lbl) for design in designs}

def get_design_hue_order(rand_mdl_names: List[str]) -> List[str]:
    """
    Order of randomization designs in seaborn plots
    Args:
        - rand_mdl_names: List of randomization design
    """

    # Benchmark designs
    complete_names = [
        name for name in rand_mdl_names if "CR" in name and "IGR" not in name
    ]
    block_names = [
        name for name in rand_mdl_names if "QB" in name and "IGR" not in name
    ]
    comp_names = [
        name for name in rand_mdl_names if "GFR" in name and "IGR" not in name
    ]

    # IGR designs (pair IGR and IGRg with same fitness function)
    restricted_name_pairs = []
    rand_mdl_names = sorted(rand_mdl_names)
    for name in rand_mdl_names:
        if "IGR" in name and "IGRg" not in name:
            igrg_name = name.replace("IGR", "IGRg")
            if igrg_name in rand_mdl_names:
                restricted_name_pairs.extend([name, igrg_name])
            else:
                restricted_name_pairs.append(name)

    ordered_names = complete_names + block_names + comp_names + restricted_name_pairs

    return ordered_names


def setup_fig(ncols: int, sharex: bool, sharey: bool, cbar: bool = False) -> tuple:
    """
    Create figure and axes for subplots
    Args:
        - ncols: Number of columns in figure
        - sharex: Share x-axis across subplots
        - sharey: Share y-axis across subplots
        - cbar: Include color
    """
    fig, axs = plt.subplots(
        1,
        ncols,
        figsize=(8 * ncols, 8),
        sharex=sharex,
        sharey=sharey,
    )
    if cbar:
        plt.subplots_adjust(wspace=0.5)
    if ncols == 1:
        axs = [axs]
    for ax in axs:
        ax.tick_params(axis="both", which="major", labelsize=16)

    return fig, axs


def format_ax(ax: plt.Axes, lbl_size:int=16):
    """
    Format axes for seaborn plots
    Args:
        - ax: Axes object
        - lbl_size: Font size for axis labels
    """
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
    ax.tick_params(axis="both", which="major", labelsize=lbl_size)

    return ax


def adjust_joint_grid_limits(jnt_grids: List[sns.JointGrid], adjust_x=True, adjust_y=True):
    """
    Adjust joint grid limits for multiple joint grids and save figures
    Args:
        - jnt_grids: List of seaborn JointGrid objects
        - adjust_x: Whether or not to adjust x-axis limits
        - adjust_y: Whether or not to adjust y-axis limits
    """
    if adjust_x:
        xmin = np.min([jnt_grid.ax_joint.get_xlim()[0] for jnt_grid in jnt_grids])
        xmax = np.max([jnt_grid.ax_joint.get_xlim()[1] for jnt_grid in jnt_grids])
        for jnt_grid in jnt_grids:
            jnt_grid.ax_joint.set_xlim(xmin, xmax)
    if adjust_y:
        y_max = np.max([jnt_grid.ax_joint.get_ylim()[1] for jnt_grid in jnt_grids])
        y_min = np.min([jnt_grid.ax_joint.get_ylim()[0] for jnt_grid in jnt_grids])
        for jnt_grid in jnt_grids:
            jnt_grid.ax_joint.set_ylim(y_min, y_max)

def save_joint_grids(jnt_grids: List[sns.JointGrid], save_dir: Path, save_fnames: List[str]):
    """
    Save joint grids as figures
    Args:
        - jnt_grids: List of seaborn JointGrid objects
        - save_dir: Directory to save figures to
        - save_fnames: List of filenames to save
    """
    for jnt_grid, save_fname in zip(jnt_grids, save_fnames):
        jnt_grid.savefig(save_dir / save_fname, transparent=True, bbox_inches="tight")
        plt.close()