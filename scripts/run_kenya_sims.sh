#!/bin/bash

declare -a params_fname_arr=(\
    "params/kenya/params_sigma_scale-0.01.csv" \
    "params/kenya/params_sigma_scale-1.0.csv")  
declare -a rand_mdl_name_arr=("restricted" "restricted-genetic")
declare -a gamma_arr=(0.5 1.0)
declare -a rep_arr=(0 1 2 3)

run_flag=$1

if [ "$run_flag" == "--make-data" ]
then
    for params_fname in "${params_fname_arr[@]}"
    do
        for gamma in "${gamma_arr[@]}"
        do
            python3 -m src.sims.run_kenya_trial \
                --make-data \
                --n-data-reps ${#rep_arr[@]} \
                --params-fname $params_fname \
                --intxn-mdl-name power-decay \
                --gamma $gamma \

        done      
    done
fi


if [ "$run_flag" == "--run-trial" ]
then
    for params_fname in "${params_fname_arr[@]}"
    do 
        for gamma in "${gamma_arr[@]}"
        do 
            for rep in "${rep_arr[@]}"
            do 
                python3 -m src.sims.run_kenya_trial \
                    --run-trial \
                    --analyze-trial \
                    --rand-mdl-name complete \
                    --cluster-lvl school \
                    --rep-to-run $rep \
                    --params-fname $params_fname \
                    --intxn-mdl-name power-decay \
                    --gamma $gamma

                python3 -m src.sims.run_kenya_trial \
                    --run-trial \
                    --analyze-trial \
                    --rand-mdl-name complete \
                    --cluster-lvl settlement \
                    --rep-to-run $rep \
                    --params-fname $params_fname \
                    --intxn-mdl-name power-decay \
                    --gamma $gamma

                for rand_mdl_name in "${rand_mdl_name_arr[@]}"
                do 
                    python3 -m src.sims.run_kenya_trial \
                        --run-trial \
                        --analyze-trial \
                        --rand-mdl-name $rand_mdl_name \
                        --cluster-lvl school \
                        --rep-to-run $rep \
                        --params-fname $params_fname \
                        --intxn-mdl-name power-decay \
                        --gamma $gamma \
                        --fitness-fn-name min-pairwise-euclidean-dist

                    python3 -m src.sims.run_kenya_trial \
                        --run-trial \
                        --analyze-trial \
                        --rand-mdl-name $rand_mdl_name \
                        --cluster-lvl school \
                        --rep-to-run $rep \
                        --params-fname $params_fname \
                        --intxn-mdl-name power-decay \
                        --gamma $gamma \
                        --fitness-fn-name frac-expo 

                    python3 -m src.sims.run_kenya_trial \
                        --run-trial \
                        --analyze-trial \
                        --rand-mdl-name $rand_mdl_name \
                        --cluster-lvl school \
                        --rep-to-run $rep \
                        --params-fname $params_fname \
                        --intxn-mdl-name power-decay \
                        --gamma $gamma \
                        --fitness-fn-name max-mahalanobis 

                    python3 -m src.sims.run_kenya_trial \
                        --run-trial \
                        --analyze-trial \
                        --rand-mdl-name $rand_mdl_name \
                        --cluster-lvl school \
                        --rep-to-run $rep \
                        --params-fname $params_fname \
                        --intxn-mdl-name power-decay \
                        --gamma $gamma \
                        --fitness-fn-name lin-comb_max-mahalanobis_euclidean-dist \
                        --fitness-fn-weights 0.5 0.5 

                    python3 -m src.sims.run_kenya_trial \
                        --run-trial \
                        --analyze-trial \
                        --rand-mdl-name $rand_mdl_name \
                        --cluster-lvl school \
                        --rep-to-run $rep \
                        --params-fname $params_fname \
                        --intxn-mdl-name power-decay \
                        --gamma $gamma \
                        --fitness-fn-name lin-comb_max-mahalanobis_frac-expo \
                        --fitness-fn-weights 0.5 0.5 
                done
            done
        done
    done

    python3 -m src.sims.collect_kenya_results
fi



