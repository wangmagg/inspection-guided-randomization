declare -a net_mdl_name_arr=("er", "sb", "ws", "ba")
declare -a rand_mdl_name_arr=("restricted" "restricted-genetic")
declare -a po_mdl_name_arr=("norm-sum")
declare -a rep_arr=(0) 

run_flag=$1

if [ "$run_flag" == "--make-data" ]
then
for net_mdl_name in "${net_mdl_name_arr[@]}"
do
    for po_mdl_name in "${po_mdl_name_arr[@]}"
    do
        python3 -m src.sims.run_network_trial \
            --make-data \
            --n-data-reps ${#rep_arr[@]} \
            --net-mdl-name $net_mdl_name \
            --potential-outcome-mdl-name $po_mdl_name
    done
done
fi

if [ "$run_flag" == "--run-trial" ]
then
    for rep in "${rep_arr[@]}"
    do
        for net_mdl_name in "${net_mdl_name_arr[@]}"
        do
            for po_mdl_name in "${po_mdl_name_arr[@]}"
            do
                python3 -m src.sims.run_network_trial \
                    --run-trial \
                    --analyze-trial \
                    --rand-mdl-name complete \
                    --net-mdl-name $net_mdl_name \
                    --potential-outcome-mdl-name $po_mdl_name \
                    --rep-to-run $rep

                python3 -m src.sims.run_network_trial \
                    --run-trial \
                    --analyze-trial \
                    --rand-mdl-name graph \
                    --net-mdl-name $net_mdl_name \
                    --potential-outcome-mdl-name $po_mdl_name \
                    --rep-to-run $rep \
                    --estimator-name clustered-diff-in-means

                for rand_mdl_name in "${rand_mdl_name_arr[@]}"
                do
                    python3 -m src.sims.run_network_trial \
                        --run-trial \
                        --analyze-trial \
                        --rand-mdl-name $rand_mdl_name \
                        --net-mdl-name $net_mdl_name \
                        --potential-outcome-mdl-name $po_mdl_name \
                        --rep-to-run $rep \
                        --fitness-fn-name lin-comb_max-mahalanobis_frac-expo \
                        --fitness-fn-weights 0 1.0

                    python3 -m src.sims.run_network_trial \
                        --run-trial \
                        --analyze-trial \
                        --rand-mdl-name $rand_mdl_name \
                        --net-mdl-name $net_mdl_name \
                        --potential-outcome-mdl-name $po_mdl_name \
                        --rep-to-run $rep \
                        --fitness-fn-name lin-comb_max-mahalanobis_frac-expo \
                        --fitness-fn-weights 0.25 0.75

                    python3 -m src.sims.run_network_trial \
                        --run-trial \
                        --analyze-trial \
                        --rand-mdl-name $rand_mdl_name \
                        --net-mdl-name $net_mdl_name \
                        --potential-outcome-mdl-name $po_mdl_name \
                        --rep-to-run $rep \
                        --fitness-fn-name lin-comb_max-mahalanobis_frac-expo \
                        --fitness-fn-weights 0.5 0.5

                    python3 -m src.sims.run_network_trial \
                        --run-trial \
                        --analyze-trial \
                        --rand-mdl-name $rand_mdl_name \
                        --net-mdl-name $net_mdl_name \
                        --potential-outcome-mdl-name $po_mdl_name \
                        --rep-to-run $rep \
                        --fitness-fn-name lin-comb_max-mahalanobis_frac-expo \
                        --fitness-fn-weights 0.75 0.25
                    
                    python3 -m src.sims.run_network_trial \
                        --run-trial \
                        --analyze-trial \
                        --rand-mdl-name $rand_mdl_name \
                        --net-mdl-name $net_mdl_name \
                        --potential-outcome-mdl-name $po_mdl_name \
                        --rep-to-run $rep \
                        --fitness-fn-name lin-comb_max-mahalanobis_frac-expo \
                        --fitness-fn-weights 1.0 0
                done
            done
        done
    done

    python3 -m src.sims.collect_results_network

fi
