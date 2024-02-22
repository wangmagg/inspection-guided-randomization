#!/bin/bash

declare -a n_arms_arr=(4 6)
declare -a restricted_rand_mdl_name_arr=("restricted" "restricted-genetic")
declare -a n_per_arm_arr=(20 30)
declare -a n_z_arr=(50000 100000)
declare -a rep_arr=(0 1 2 3)

run_flag=$1

if [ "$run_flag" == "--make-data" ]
then
    for n_arms in "${n_arms_arr[@]}"
    do
        for n_per_arm in "${n_per_arm_arr[@]}"
        do
            python3 -m src.sims.run_multarm_trial \
                --exp-dir mult-arm-imbalance \
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
            for rep in "${rep_arr[@]}"
            do
                for n_z in "${n_z_arr[@]}"
                do
                    for rand_mdl_name in "${restricted_rand_mdl_name_arr[@]}"
                    do 
                        if [ $n_arms -eq 4 ]
                        then
                            python3 -m src.sims.run_multarm_trial \
                                --exp-dir mult-arm-imbalance \
                                --n-arms $n_arms \
                                --n-per-arm $n_per_arm \
                                --n-z $n_z \
                                --rand-mdl-name $rand_mdl_name \
                                --rep-to-run $rep \
                                --fitness-fn-name weighted-sum-smd \
                                --fitness-fn-weights 1 1 6 6 \
                                --run-trial

                            python3 -m src.sims.run_multarm_trial \
                                --exp-dir mult-arm-imbalance \
                                --n-arms $n_arms \
                                --n-per-arm $n_per_arm \
                                --n-z $n_z \
                                --rand-mdl-name $rand_mdl_name \
                                --rep-to-run $rep \
                                --fitness-fn-name weighted-sum-smd \
                                --fitness-fn-weights 1 1 12 12 \
                                --run-trial
                        fi

                        if [ $n_arms -eq 6 ]
                        then
                            python3 -m src.sims.run_multarm_trial \
                                --exp-dir mult-arm-imbalance \
                                --n-arms $n_arms \
                                --n-per-arm $n_per_arm \
                                --n-z $n_z \
                                --rand-mdl-name $rand_mdl_name \
                                --rep-to-run $rep \
                                --fitness-fn-name weighted-sum-smd \
                                --fitness-fn-weights 1 1 -6 -6 6 6 \
                                --run-trial

                            python3 -m src.sims.run_multarm_trial \
                                --exp-dir mult-arm-imbalance \
                                --n-arms $n_arms \
                                --n-per-arm $n_per_arm \
                                --n-z $n_z \
                                --rand-mdl-name $rand_mdl_name \
                                --rep-to-run $rep \
                                --fitness-fn-name weighted-sum-smd \
                                --fitness-fn-weights 1 1 -12 -12 12 12 \
                                --run-trial
                        fi
                    done
                done
            done
        done
    done

    python3 -m src.sims.collect_multarm_results \
        --exp-subdir mult-arm-imbalance

fi



