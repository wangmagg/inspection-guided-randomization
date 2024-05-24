#!/bin/bash

declare -a rand_mdl_name_arr=("restricted" "restricted-genetic")
declare -a rep_arr=(0)
declare -a n_z_lg_arr=(500000 250000)
# declare -a n_z_arr=(500000 250000 150000 90000 80000 70000 60000 50000)
declare -a seed_arr=(42 0 1 2 3 4)

for seed in "${seed_arr[@]}"
do
    for n_z in "${n_z_lg_arr[@]}"
    do
        python3 -m src.sims.run_kenya_trial \
            --run-trial \
            --analyze-trial \
            --seed $seed \
            --n-z $n_z \
            --n-batches 10 \
            --param-fname params/ki-mu-pos_ko-da-pos_sd-sm_bal.csv \
            --potential-outcome-mdl-name kenya-hierarchical-nbr-sum \
            --rand-mdl-name complete \
            --cluster-lvl school \
            --net-mdl-name nested-2lvl-sb \
            --rep-to-run 0 \
            --intxn-mdl-name power-decay \
            --gamma 0.5 \
            --p-same-in 0.20 \
            --p-diff-in-same-out 0.02 \
            --p-diff-in-diff-out 0.01 

        for rand_mdl_name in "${rand_mdl_name_arr[@]}"
        do 
            python3 -m src.sims.run_kenya_trial \
                --run-trial \
                --analyze-trial \
                --seed $seed \
                --n-z $n_z \
                --n-batches 10 \
                --param-fname params/ki-mu-pos_ko-da-pos_sd-sm_bal.csv \
                --potential-outcome-mdl-name kenya-hierarchical-nbr-sum \
                --rand-mdl-name $rand_mdl_name \
                --net-mdl-name nested-2lvl-sb \
                --cluster-lvl school \
                --rep-to-run 0 \
                --intxn-mdl-name power-decay \
                --gamma 0.5 \
                --p-same-in 0.20 \
                --p-diff-in-same-out 0.02 \
                --p-diff-in-diff-out 0.01 \
                --fitness-fn-name lin-comb_max-mahalanobis_frac-expo \
                --fitness-fn-weights 0.5 0.5 

            python3 -m src.sims.run_trial_kenya \
                --run-trial \
                --analyze-trial \
                --seed $seed \
                --n-z $n_z \
                --n-batches 10 \
                --param-fname params/ki-mu-pos_ko-da-pos_sd-sm_bal.csv \
                --potential-outcome-mdl-name kenya-hierarchical-nbr-sum \
                --rand-mdl-name $rand_mdl_name \
                --add-all-mirrors \
                --net-mdl-name nested-2lvl-sb \
                --cluster-lvl school \
                --rep-to-run 0 \
                --intxn-mdl-name power-decay \
                --gamma 0.5 \
                --p-same-in 0.20 \
                --p-diff-in-same-out 0.02 \
                --p-diff-in-diff-out 0.01 \
                --fitness-fn-name lin-comb_max-mahalanobis_frac-expo \
                --fitness-fn-weights 0.5 0.5 
        done
    done
    python3 -m src.sims.collect_results_kenya --exp-subdir kenya-pool
done