#!/bin/bash

declare -a n_arms_arr=(4 6)
declare -a restricted_rand_mdl_name_arr=("restricted" "restricted-genetic")
declare -a n_per_arm_arr=(20 30)
declare -a n_z_arr=(50000 100000)
declare -a rep_arr=(0)


run_flag=$1

if [ "$run_flag" == "--make-data" ]
then
    for n_arms in "${n_arms_arr[@]}"
    do
        for n_per_arm in "${n_per_arm_arr[@]}"
        do
            python3 -m src.sims.run_multarm_trial \
                --n-arms $n_arms \
                --n-per-arm $n_per_arm \
                --n-data-reps ${#rep_arr[@]} \
                --make-data
        done
    done
fi

if [ "$run_flag" == "--run-trial" ]
then 
    for n_arms in "${n_arms_arr[@]}"
    do 
        for n_per_arm in "${n_per_arm_arr[@]}"
        do
            Rscript src/design/randomization_quick_block.R \
                --n-arms $n_arms \
                --n-per-arm $n_per_arm \
                --n-data-reps ${#rep_arr[@]}\
                --min-block-factor 2 

            for rep in "${rep_arr[@]}"
            do
                python3 -m src.sims.run_multarm_trial \
                    --n-arms $n_arms \
                    --n-per-arm $n_per_arm \
                    --rand-mdl-name quick-block \
                    --rep-to-run $rep \
                    --min-block-factor 2 \
                    --estimator-name qb-diff-in-means \
                    --run-trial \
                    --analyze-trial

                for n_z in "${n_z_arr[@]}"
                do
                    python3 -m src.sims.run_multarm_trial \
                    --n-arms $n_arms \
                    --n-per-arm $n_per_arm \
                    --n-z $n_z \
                    --rand-mdl-name complete \
                    --rep-to-run $rep \
                    --run-trial \
                    --analyze-trial

                    for rand_mdl_name in "${restricted_rand_mdl_name_arr[@]}"
                    do 
                        python3 -m src.sims.run_multarm_trial \
                            --n-arms $n_arms \
                            --n-per-arm $n_per_arm \
                            --n-z $n_z \
                            --rand-mdl-name $rand_mdl_name \
                            --rep-to-run $rep \
                            --fitness-fn-name sum-max-abs-smd \
                            --run-trial \
                            --analyze-trial

                        python3 -m src.sims.run_multarm_trial \
                            --n-arms $n_arms \
                            --n-per-arm $n_per_arm \
                            --n-z $n_z \
                            --rand-mdl-name $rand_mdl_name \
                            --rep-to-run $rep \
                            --fitness-fn-name max-mahalanobis \
                            --run-trial \
                            --analyze-trial
                    done
                done
            done
        done
    done

    python3 -m src.sims.collect_results_multarm
fi


