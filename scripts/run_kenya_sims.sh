#!/bin/bash
declare -a param_fname_arr=( "params/ki-mu-pos_ko-da-pos_sd-sm_bal.csv")   
declare -a net_mdl_name_arr=("nested-2lvl-sb")
declare -a rand_mdl_name_arr=("restricted" "restricted-genetic")
declare -a po_mdl_name_arr=("kenya-hierarchical-nbr-sum")
declare -a gamma_arr=(0.5 0.25 0.75)
declare -a rep_arr=(0)

run_flag=$1
p_same_in=$2
p_diff_in_same_out=$3
p_diff_in_diff_out=$4

if [ "$run_flag" == "--make-data" ]
then
    for net_mdl_name in "${net_mdl_name_arr[@]}"
    do
        for gamma in "${gamma_arr[@]}"
        do
            for param_fname in "${param_fname_arr[@]}"
            do
                for po_mdl_name in "${po_mdl_name_arr[@]}"
                do
                    python3 -m src.sims.run_kenya_trial \
                        --make-data \
                        --param-fname $param_fname \
                        --n-data-reps ${#rep_arr[@]} \
                        --potential-outcome-mdl-name $po_mdl_name \
                        --net-mdl-name $net_mdl_name \
                        --intxn-mdl-name power-decay \
                        --gamma $gamma \
                        --p-same-in $p_same_in \
                        --p-diff-in-same-out $p_diff_in_same_out \
                        --p-diff-in-diff-out $p_diff_in_diff_out 
                done      
            done
        done
    done
fi


if [ "$run_flag" == "--run-trial" ]
then
    for rep in "${rep_arr[@]}"
    do 
        for net_mdl_name in "${net_mdl_name_arr[@]}"
        do 
            for gamma in "${gamma_arr[@]}"
            do 
                for param_fname in "${param_fname_arr[@]}"
                do
                    for po_mdl_name in "${po_mdl_name_arr[@]}"
                    do
                        python3 -m src.sims.run_kenya_trial \
                            --run-trial \
                            --analyze-trial \
                            --param-fname $param_fname \
                            --potential-outcome-mdl-name $po_mdl_name \
                            --rand-mdl-name complete \
                            --cluster-lvl school \
                            --net-mdl-name $net_mdl_name \
                            --rep-to-run $rep \
                            --intxn-mdl-name power-decay \
                            --gamma $gamma \
                            --p-same-in $p_same_in \
                            --p-diff-in-same-out $p_diff_in_same_out \
                            --p-diff-in-diff-out $p_diff_in_diff_out

                        python3 -m src.sims.run_kenya_trial \
                            --run-trial \
                            --analyze-trial \
                            --param-fname $param_fname \
                            --potential-outcome-mdl-name $po_mdl_name \
                            --rand-mdl-name complete \
                            --net-mdl-name $net_mdl_name \
                            --cluster-lvl settlement \
                            --rep-to-run $rep \
                            --intxn-mdl-name power-decay \
                            --gamma $gamma \
                            --p-same-in $p_same_in \
                            --p-diff-in-same-out $p_diff_in_same_out \
                            --p-diff-in-diff-out $p_diff_in_diff_out

                        for rand_mdl_name in "${rand_mdl_name_arr[@]}"
                        do 
                            # add all mirror allocations
                            python3 -m src.sims.run_kenya_trial \
                                --run-trial \
                                --analyze-trial \
                                --param-fname $param_fname \
                                --potential-outcome-mdl-name $po_mdl_name \
                                --rand-mdl-name $rand_mdl_name \
                                --net-mdl-name $net_mdl_name \
                                --cluster-lvl school \
                                --rep-to-run $rep \
                                --intxn-mdl-name power-decay \
                                --gamma $gamma \
                                --p-same-in $p_same_in \
                                --p-diff-in-same-out $p_diff_in_same_out \
                                --p-diff-in-diff-out $p_diff_in_diff_out \
                                --fitness-fn-name lin-comb_max-mahalanobis_frac-expo \
                                --fitness-fn-weights 0.5 0.5

                            python3 -m src.sims.run_kenya_trial \
                                --run-trial \
                                --analyze-trial \
                                --param-fname $param_fname \
                                --potential-outcome-mdl-name $po_mdl_name \
                                --rand-mdl-name $rand_mdl_name \
                                --net-mdl-name $net_mdl_name \
                                --cluster-lvl school \
                                --rep-to-run $rep \
                                --intxn-mdl-name power-decay \
                                --gamma $gamma \
                                --p-same-in $p_same_in \
                                --p-diff-in-same-out $p_diff_in_same_out \
                                --p-diff-in-diff-out $p_diff_in_diff_out \
                                --fitness-fn-name frac-expo 

                            python3 -m src.sims.run_kenya_trial \
                                --run-trial \
                                --analyze-trial \
                                --param-fname $param_fname \
                                --potential-outcome-mdl-name $po_mdl_name \
                                --rand-mdl-name $rand_mdl_name \
                                --net-mdl-name $net_mdl_name \
                                --cluster-lvl school \
                                --rep-to-run $rep \
                                --intxn-mdl-name power-decay \
                                --gamma $gamma \
                                --p-same-in $p_same_in \
                                --p-diff-in-same-out $p_diff_in_same_out \
                                --p-diff-in-diff-out $p_diff_in_diff_out \
                                --fitness-fn-name lin-comb_max-mahalanobis_frac-expo \
                                --fitness-fn-weights 0.25 0.75

                            python3 -m src.sims.run_kenya_trial \
                                --run-trial \
                                --analyze-trial \
                                --param-fname $param_fname \
                                --potential-outcome-mdl-name $po_mdl_name \
                                --rand-mdl-name $rand_mdl_name \
                                --net-mdl-name $net_mdl_name \
                                --cluster-lvl school \
                                --rep-to-run $rep \
                                --intxn-mdl-name power-decay \
                                --gamma $gamma \
                                --p-same-in $p_same_in \
                                --p-diff-in-same-out $p_diff_in_same_out \
                                --p-diff-in-diff-out $p_diff_in_diff_out \
                                --fitness-fn-name lin-comb_max-mahalanobis_frac-expo \
                                --fitness-fn-weights 0.75 0.25

                            python3 -m src.sims.run_kenya_trial \
                                --run-trial \
                                --analyze-trial \
                                --param-fname $param_fname \
                                --potential-outcome-mdl-name $po_mdl_name \
                                --rand-mdl-name $rand_mdl_name \
                                --net-mdl-name $net_mdl_name \
                                --cluster-lvl school \
                                --rep-to-run $rep \
                                --intxn-mdl-name power-decay \
                                --gamma $gamma \
                                --p-same-in $p_same_in \
                                --p-diff-in-same-out $p_diff_in_same_out \
                                --p-diff-in-diff-out $p_diff_in_diff_out \
                                --fitness-fn-name max-mahalanobis 

                            python3 -m src.sims.run_kenya_trial \
                                --run-trial \
                                --analyze-trial \
                                --param-fname $param_fname \
                                --potential-outcome-mdl-name $po_mdl_name \
                                --rand-mdl-name $rand_mdl_name \
                                --net-mdl-name $net_mdl_name \
                                --cluster-lvl school \
                                --rep-to-run $rep \
                                --intxn-mdl-name power-decay \
                                --gamma $gamma \
                                --p-same-in $p_same_in \
                                --p-diff-in-same-out $p_diff_in_same_out \
                                --p-diff-in-diff-out $p_diff_in_diff_out \
                                --fitness-fn-name lin-comb_max-mahalanobis_euclidean-dist \
                                --fitness-fn-weights 0.5 0.5

                            python3 -m src.sims.run_kenya_trial \
                                --run-trial \
                                --analyze-trial \
                                --param-fname $param_fname \
                                --potential-outcome-mdl-name $po_mdl_name \
                                --rand-mdl-name $rand_mdl_name \
                                --net-mdl-name $net_mdl_name \
                                --cluster-lvl school \
                                --rep-to-run $rep \
                                --intxn-mdl-name power-decay \
                                --gamma $gamma \
                                --p-same-in $p_same_in \
                                --p-diff-in-same-out $p_diff_in_same_out \
                                --p-diff-in-diff-out $p_diff_in_diff_out \
                                --fitness-fn-name min-pairwise-euclidean-dist

                            python3 -m src.sims.run_kenya_trial \
                                --run-trial \
                                --analyze-trial \
                                --param-fname $param_fname \
                                --potential-outcome-mdl-name $po_mdl_name \
                                --rand-mdl-name $rand_mdl_name \
                                --net-mdl-name $net_mdl_name \
                                --cluster-lvl school \
                                --rep-to-run $rep \
                                --intxn-mdl-name power-decay \
                                --gamma $gamma \
                                --p-same-in $p_same_in \
                                --p-diff-in-same-out $p_diff_in_same_out \
                                --p-diff-in-diff-out $p_diff_in_diff_out \
                                --fitness-fn-name lin-comb_max-mahalanobis_euclidean-dist \
                                --fitness-fn-weights 0.75 0.25

                            python3 -m src.sims.run_kenya_trial \
                                --run-trial \
                                --analyze-trial \
                                --param-fname $param_fname \
                                --potential-outcome-mdl-name $po_mdl_name \
                                --rand-mdl-name $rand_mdl_name \
                                --net-mdl-name $net_mdl_name \
                                --cluster-lvl school \
                                --rep-to-run $rep \
                                --intxn-mdl-name power-decay \
                                --gamma $gamma \
                                --p-same-in $p_same_in \
                                --p-diff-in-same-out $p_diff_in_same_out \
                                --p-diff-in-diff-out $p_diff_in_diff_out \
                                --fitness-fn-name lin-comb_max-mahalanobis_euclidean-dist \
                                --fitness-fn-weights 0.25 0.75
                            
                            # add mirror allocations only if they are not worse
                            python3 -m src.sims.run_kenya_trial \
                                --run-trial \
                                --analyze-trial \
                                --param-fname $param_fname \
                                --potential-outcome-mdl-name $po_mdl_name \
                                --rand-mdl-name $rand_mdl_name \
                                --add-all-mirrors \
                                --net-mdl-name $net_mdl_name \
                                --cluster-lvl school \
                                --rep-to-run $rep \
                                --intxn-mdl-name power-decay \
                                --gamma $gamma \
                                --p-same-in $p_same_in \
                                --p-diff-in-same-out $p_diff_in_same_out \
                                --p-diff-in-diff-out $p_diff_in_diff_out \
                                --fitness-fn-name lin-comb_max-mahalanobis_frac-expo \
                                --fitness-fn-weights 0.5 0.5

                            python3 -m src.sims.run_kenya_trial \
                                --run-trial \
                                --analyze-trial \
                                --param-fname $param_fname \
                                --potential-outcome-mdl-name $po_mdl_name \
                                --rand-mdl-name $rand_mdl_name \
                                --add-all-mirrors \
                                --net-mdl-name $net_mdl_name \
                                --cluster-lvl school \
                                --rep-to-run $rep \
                                --intxn-mdl-name power-decay \
                                --gamma $gamma \
                                --p-same-in $p_same_in \
                                --p-diff-in-same-out $p_diff_in_same_out \
                                --p-diff-in-diff-out $p_diff_in_diff_out \
                                --fitness-fn-name frac-expo 

                            python3 -m src.sims.run_kenya_trial \
                                --run-trial \
                                --analyze-trial \
                                --param-fname $param_fname \
                                --potential-outcome-mdl-name $po_mdl_name \
                                --rand-mdl-name $rand_mdl_name \
                                --add-all-mirrors \
                                --net-mdl-name $net_mdl_name \
                                --cluster-lvl school \
                                --rep-to-run $rep \
                                --intxn-mdl-name power-decay \
                                --gamma $gamma \
                                --p-same-in $p_same_in \
                                --p-diff-in-same-out $p_diff_in_same_out \
                                --p-diff-in-diff-out $p_diff_in_diff_out \
                                --fitness-fn-name lin-comb_max-mahalanobis_frac-expo \
                                --fitness-fn-weights 0.25 0.75

                            python3 -m src.sims.run_kenya_trial \
                                --run-trial \
                                --analyze-trial \
                                --param-fname $param_fname \
                                --potential-outcome-mdl-name $po_mdl_name \
                                --rand-mdl-name $rand_mdl_name \
                                --add-all-mirrors \
                                --net-mdl-name $net_mdl_name \
                                --cluster-lvl school \
                                --rep-to-run $rep \
                                --intxn-mdl-name power-decay \
                                --gamma $gamma \
                                --p-same-in $p_same_in \
                                --p-diff-in-same-out $p_diff_in_same_out \
                                --p-diff-in-diff-out $p_diff_in_diff_out \
                                --fitness-fn-name lin-comb_max-mahalanobis_frac-expo \
                                --fitness-fn-weights 0.75 0.25

                            python3 -m src.sims.run_kenya_trial \
                                --run-trial \
                                --analyze-trial \
                                --param-fname $param_fname \
                                --potential-outcome-mdl-name $po_mdl_name \
                                --rand-mdl-name $rand_mdl_name \
                                --add-all-mirrors \
                                --net-mdl-name $net_mdl_name \
                                --cluster-lvl school \
                                --rep-to-run $rep \
                                --intxn-mdl-name power-decay \
                                --gamma $gamma \
                                --p-same-in $p_same_in \
                                --p-diff-in-same-out $p_diff_in_same_out \
                                --p-diff-in-diff-out $p_diff_in_diff_out \
                                --fitness-fn-name max-mahalanobis 

                            python3 -m src.sims.run_kenya_trial \
                                --run-trial \
                                --analyze-trial \
                                --param-fname $param_fname \
                                --potential-outcome-mdl-name $po_mdl_name \
                                --rand-mdl-name $rand_mdl_name \
                                --add-all-mirrors \
                                --net-mdl-name $net_mdl_name \
                                --cluster-lvl school \
                                --rep-to-run $rep \
                                --intxn-mdl-name power-decay \
                                --gamma $gamma \
                                --p-same-in $p_same_in \
                                --p-diff-in-same-out $p_diff_in_same_out \
                                --p-diff-in-diff-out $p_diff_in_diff_out \
                                --fitness-fn-name lin-comb_max-mahalanobis_euclidean-dist \
                                --fitness-fn-weights 0.5 0.5

                            python3 -m src.sims.run_kenya_trial \
                                --run-trial \
                                --analyze-trial \
                                --param-fname $param_fname \
                                --potential-outcome-mdl-name $po_mdl_name \
                                --rand-mdl-name $rand_mdl_name \
                                --add-all-mirrors \
                                --net-mdl-name $net_mdl_name \
                                --cluster-lvl school \
                                --rep-to-run $rep \
                                --intxn-mdl-name power-decay \
                                --gamma $gamma \
                                --p-same-in $p_same_in \
                                --p-diff-in-same-out $p_diff_in_same_out \
                                --p-diff-in-diff-out $p_diff_in_diff_out \
                                --fitness-fn-name min-pairwise-euclidean-dist

                            python3 -m src.sims.run_kenya_trial \
                                --run-trial \
                                --analyze-trial \
                                --param-fname $param_fname \
                                --potential-outcome-mdl-name $po_mdl_name \
                                --rand-mdl-name $rand_mdl_name \
                                --add-all-mirrors \
                                --net-mdl-name $net_mdl_name \
                                --cluster-lvl school \
                                --rep-to-run $rep \
                                --intxn-mdl-name power-decay \
                                --gamma $gamma \
                                --p-same-in $p_same_in \
                                --p-diff-in-same-out $p_diff_in_same_out \
                                --p-diff-in-diff-out $p_diff_in_diff_out \
                                --fitness-fn-name lin-comb_max-mahalanobis_euclidean-dist \
                                --fitness-fn-weights 0.75 0.25

                            python3 -m src.sims.run_kenya_trial \
                                --run-trial \
                                --analyze-trial \
                                --param-fname $param_fname \
                                --potential-outcome-mdl-name $po_mdl_name \
                                --rand-mdl-name $rand_mdl_name \
                                --add-all-mirrors \
                                --net-mdl-name $net_mdl_name \
                                --cluster-lvl school \
                                --rep-to-run $rep \
                                --intxn-mdl-name power-decay \
                                --gamma $gamma \
                                --p-same-in $p_same_in \
                                --p-diff-in-same-out $p_diff_in_same_out \
                                --p-diff-in-diff-out $p_diff_in_diff_out \
                                --fitness-fn-name lin-comb_max-mahalanobis_euclidean-dist \
                                --fitness-fn-weights 0.25 0.75
                        done
                    done
                done
            done
        done
    done
    python3 -m src.sims.collect_results_kenya
fi



