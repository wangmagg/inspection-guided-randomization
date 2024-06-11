declare -a net_mdl_name_arr=("er", "sb", "ws", "ba")
declare -a rand_mdl_name_arr=("restricted" "restricted-genetic")
declare -a po_mdl_name_arr=("norm-sum")
declare -a p_misspec_arr=(0.01 0.03 0.06 0.10 0.15 0.21 0.28 0.36 0.45)
declare -a seed_arr=(42 0 1 2 3 4)

for seed in "${seed_arr[@]}"
do  
    for net_mdl_name in "${net_mdl_name_arr[@]}"
    do
        for rand_mdl_name in "${rand_mdl_name_arr[@]}"
        do
            for p_misspec in "${p_misspec_arr[@]}"
            do
                python3 -m src.sims.run_network_trial \
                    --run-trial \
                    --analyze-trial \
                    --output-subdir network-misspec \
                    --rand-mdl-name $rand_mdl_name \
                    --net-mdl-name $net_mdl_name \
                    --potential-outcome-mdl-name norm-sum \
                    --rep-to-run 0 \
                    --fitness-fn-name lin-comb_max-mahalanobis_frac-expo \
                    --fitness-fn-weights 0.5 0.5 \
                    --p-misspec $p_misspec \
                    --misspec-seed $seed

                python3 -m src.sims.run_network_trial \
                    --run-trial \
                    --analyze-trial \
                    --output-subdir network-misspec \
                    --rand-mdl-name $rand_mdl_name \
                    --add-all-mirrors \
                    --net-mdl-name $net_mdl_name \
                    --potential-outcome-mdl-name norm-sum \
                    --rep-to-run 0 \
                    --fitness-fn-name lin-comb_max-mahalanobis_frac-expo \
                    --fitness-fn-weights 0.5 0.5 \
                    --p-misspec $p_misspec \
                    --misspec-seed $seed
                done
            done
        done
    done

    python3 -m src.sims.collect_results_network --output-subdir network-misspec

done

