#!/bin/bash

python3 -m src.visualization.make_plots \
    --exp-subdir mult-arm \
    --fig-types cov_pairwise_balance \
    --balance-fn smd

python3 -m src.visualization.make_plots \
    --exp-subdir mult-arm \
    --fig-types cov_balance \
    --balance-fn signed-max-abs-smd

python3 -m src.visualization.make_plots \
    --exp-subdir mult-arm \
    --fig-types scatter_error_vs_score \

python3 -m src.visualization.make_plots \
    --exp-subdir mult-arm \
    --fig-types hist_same_z_validity hist_z 
