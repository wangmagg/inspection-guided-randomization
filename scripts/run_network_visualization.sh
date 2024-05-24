#!/bin/bash

python3 -m src.visualization.make_plots \
    --exp-subdir network \
    --fig-types hist_2dscore_disagg hist_same_z_validity hist_z 

python3 -m src.visualization.make_plots \
    --exp-subdir network \
    --fig-types scatter_error_vs_score

python3 -m src.visualization.make_plots \
    --exp-subdir network \
    --fig-types scatter_2dscore_disagg \
    --axis-fns max-mahalanobis frac-expo

    