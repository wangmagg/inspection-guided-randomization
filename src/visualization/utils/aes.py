import numpy as np
from typing import List
from matplotlib.ticker import FormatStrFormatter

def get_hue_order(rand_mdl_names, ff_in_name=True):
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
    rand_mdl_names = np.sort(rand_mdl_names)
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

    if not ff_in_name:
        ordered_names = [name.split("-")[0].strip() for name in ordered_names]

    return ordered_names


def color_mapping(rand_mdl_name):
    """
    Map mult-arm randomization design names to colors
    """

    # Benchmarks are blue scheme
    if "CR" in rand_mdl_name and "Set" not in rand_mdl_name:
        return "navy"
    if "CR" in rand_mdl_name and "Set" in rand_mdl_name:
        return "grey"
    if rand_mdl_name == "QB" or rand_mdl_name == "GR" or rand_mdl_name == "GFR":
        return "cornflowerblue"

    if (
        "Mahalanobis" in rand_mdl_name
        and "IGR" in rand_mdl_name
        and "IGRg" not in rand_mdl_name
        and "+" not in rand_mdl_name
    ):
        return "lightsalmon"
    if (
        "Mahalanobis" in rand_mdl_name
        and "IGRg" in rand_mdl_name
        and "+" not in rand_mdl_name
    ):
        return "orangered"
    if (
        "SumMaxAbsSMD" in rand_mdl_name
        and "IGR" in rand_mdl_name
        and "IGRg" not in rand_mdl_name
    ):
        return "gold"
    if "SumMaxAbsSMD" in rand_mdl_name and "IGRg" in rand_mdl_name:
        return "darkorange"

    if "IGR" in rand_mdl_name and "IGRg" not in rand_mdl_name:
        if "FracCtrlExposed" in rand_mdl_name:
            return "darkseagreen"
        elif "Dist" in rand_mdl_name:
            return "thistle"
    if "IGRg" in rand_mdl_name:
        if "FracCtrlExposed" in rand_mdl_name:
            return "seagreen"
        elif "Dist" in rand_mdl_name:
            return "mediumorchid"


def two_tone_color_mapping(rand_mdl_name):
    if r"$\mathcal{Z}_{pool}$" in rand_mdl_name:
        return "silver"
    elif r"$\mathcal{Z}^{*}_{pool}$" in rand_mdl_name:
        return "dimgrey"

def get_palette(rand_mdl_names, ff_in_name=True):
    """
    Create palette dictionary mapping randomization design names to colors
    """

    if ff_in_name:
        return {
            rand_mdl_name: color_mapping(rand_mdl_name)
            for rand_mdl_name in rand_mdl_names
        }
    else:
        rand_mdl_names_no_ff = [name.split("-")[0].strip() for name in rand_mdl_names]
        return {
            rand_mdl_name_no_ff: color_mapping(rand_mdl_name)
            for (rand_mdl_name_no_ff, rand_mdl_name) in zip(
                rand_mdl_names_no_ff, rand_mdl_names
            )
        }

def get_two_tone_palette(enum_types):
    return {
        enum_type: two_tone_color_mapping(enum_type)
        for enum_type in enum_types
    }

def format_ax(ax):
    max_x = ax.get_xlim()[1]
    max_y = ax.get_ylim()[1]

    if max_x < 0.1:
        x_fmt = "%.3f"
    else:
        x_fmt = "%.2f"
    if max_y < 0.1:
        y_fmt = "%.3f"
    else:
        y_fmt = "%.2f"

    ax.xaxis.set_major_formatter(FormatStrFormatter(x_fmt))
    ax.yaxis.set_major_formatter(FormatStrFormatter(y_fmt))
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    return ax

def axis_lim(df_list, col_name_list):
    if not isinstance(col_name_list, List):
        min_val = np.min([df[col_name_list].min() for df in df_list])
        max_val = np.max([df[col_name_list].max() for df in df_list])
    else:
        min_val = np.min(
                [
                    df[col_name].min()
                    for df, col_name in zip(df_list, col_name_list)
                ]
            )
        max_val = np.max(
                [
                    df[col_name].max()
                    for df, col_name in zip(df_list, col_name_list)
                ]
            )
    return min_val, max_val