#!/bin/bash

declare -a param_subdirs_arr=(\
    "params_sigma_scale-1.0"\
    "params_sigma_scale-0.01")

for param_subdir in "${param_subdirs_arr[@]}"
do
    python3 -m src.visualization.make_plots \
        --exp-subdir kenya \
        --param-subdir $param_subdir \
        --fig-types "scatter_error_vs_score" \
        --axis-fns "max-mahalanobis" "frac-expo" 

    python3 -m src.visualization.make_plots \
        --exp-subdir kenya \
        --param-subdir $param_subdir \
        --fig-types "hist_same_z_validity" "hist_z" 

    python3 -m src.visualization.make_plots \
        --exp-subdir kenya \
        --param-subdir $param_subdir \
        --fig-types "scatter_2dscore_disagg" \
        --axis-fns "max-mahalanobis" "frac-expo" 

    python3 -m src.visualization.make_kenya_plots \
        --exp-subdir kenya \
        --param-subdir $param_subdir \
        --fig-types "school_locs" "covariate_distr_across_sets" "adjacency" "adj_v_dist"
done

