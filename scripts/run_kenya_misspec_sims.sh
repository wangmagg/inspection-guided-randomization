#!/bin/bash

declare -a rand_mdl_name_arr=("restricted" "restricted-genetic")
declare -a p_misspec_arr=(0.01 0.03 0.06 0.10 0.15 0.21 0.28 0.36 0.45)
declare -a seed_arr=(0 1 2 3 4)

for seed in "${seed_arr[@]}"
do  
    for rand_mdl_name in "${rand_mdl_name_arr[@]}"
    do
        for p_misspec in "${p_misspec_arr[@]}"
        do
            python3 -m src.sims.run_kenya_trial \
                --run-trial \
                --analyze-trial \
                --output-subdir kenya-misspec \
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
                --fitness-fn-weights 0.5 0.5 \
                --n-cutoff 500 \
                --p-misspec $p_misspec \
                --misspec-seed $seed

            python3 -m src.sims.run_kenya_trial \
                --run-trial \
                --analyze-trial \
                --output-subdir kenya-misspec \
                --add-all-mirrors \
                --param-fname params/ki-mu-pos_ko-da-pos_sd-sm_bal.csv \
                --potential-outcome-mdl-name kenya-hierarchical-nbr-sum \
                --rand-mdl-name $rand_mdl_name \
                --net-mdl-name "nested-2lvl-sb" \
                --cluster-lvl school \
                --rep-to-run 0 \
                --intxn-mdl-name power-decay \
                --gamma 0.5 \
                --p-same-in 0.20 \
                --p-diff-in-same-out 0.02 \
                --p-diff-in-diff-out 0.01 \
                --fitness-fn-name lin-comb_max-mahalanobis_frac-expo \
                --fitness-fn-weights 0.5 0.5 \
                --n-cutoff 500 \
                --p-misspec $p_misspec \
                --misspec-seed $seed
        done
    done
    python3 -m src.sims.collect_results_kenya --exp-subdir kenya-misspec
done

