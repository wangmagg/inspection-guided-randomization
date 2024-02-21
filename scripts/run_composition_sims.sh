#!/bin/bash

declare -a n_per_arm_arr=(20 30)
declare -a n_z_arr=(50000 100000)
declare -a rep_arr=(0 1 2 3)

run_flag=$1

if [ "$run_flag" == "--make-data" ]
then
    for n_per_arm in "${n_per_arm_arr[@]}"
    do
        python3 -m src.sims.run_composition_trial \
            --n-per-arm $n_per_arm \
            --n-data-reps ${#rep_arr[@]} \
            --p-comps 0.5 0.3 0.7 \
            --make-data
    done
fi

if [ "$run_flag" == "--run-trial" ]
then
    for n_per_arm in "${n_per_arm_arr[@]}"
    do
        for rep in "${rep_arr[@]}"
        do
            python3 -m src.sims.run_composition_trial \
                --n-per-arm $n_per_arm \
                --rep-to-run $rep \
                --rand-mdl-name group-formation \
                --run-trial \
                --analyze-trial

            python3 -m src.sims.run_composition_trial \
                --n-per-arm $n_per_arm \
                --rep-to-run $rep \
                --rand-mdl-name group-formation-restricted \
                --fitness-fn-name max-mahalanobis \
                --run-trial \
                --analyze-trial

            python3 -m src.sims.run_composition_trial \
                --n-per-arm $n_per_arm \
                --rep-to-run $rep \
                --rand-mdl-name group-formation-restricted \
                --fitness-fn-name sum-max-smd \
                --run-trial \
                --analyze-trial
        done
    done

    python3 -m src.sims.collect_composition_results
fi