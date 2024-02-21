#!/bin/bash

#!/bin/bash

python3 -m src.visualization.make_plots \
    --exp-subdir mult-arm-imbalance \
    --fig-types scatter_error_vs_score \

python3 -m src.visualization.make_plots \
    --exp-subdir mult-arm-imbalance \
    --fig-types cov_pairwise_balance \
    --balance-fn smd

python3 -m src.visualization.make_plots \
    --exp-subdir mult-arm-imbalance \
    --fig-types cov_balance \
    --balance-fn signed-max-abs-smd
