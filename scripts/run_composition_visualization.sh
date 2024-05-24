#!/bin/bash

python3 -m src.visualization.make_plots \
    --exp-subdir composition \
    --fig-types hist_same_z_validity hist_z 
    
python3 -m src.visualization.make_plots \
    --exp-subdir composition \
    --fig-types cov_distr cov_pairwise_balance \
    --balance-fn smd

python3 -m src.visualization.make_plots \
    --exp-subdir composition \
    --fig-types cov_balance \
    --balance-fn signed-max-abs-smd


