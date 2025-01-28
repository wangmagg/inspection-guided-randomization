#!/bin/bash

declare -a iter_arr=(0 1 2 3 4)
declare -a enum_arr=(5000 10000 25000 50000 100000 200000) 
declare -a accept_arr=(100 500 1000)  
declare -a mirror_arr=("all" "none")
res_dir="res_resub/interference"

for iter in "${iter_arr[@]}"
do
    for mirror in "${mirror_arr[@]}"
    do
        python3 -m vignettes.interference \
            --data-iter $iter \
            --n-enum 100000 \
            --n-accept 500 \
            --mirror-type $mirror \
            --tau-size 0.3 \
            --gamma 0.5 \
            --w1 0.25 0.5 \
            --w2 0.75 0.5 \
            --out-dir $res_dir \
            --save-checks-figs

        python3 -m vignettes.interference \
            --data-iter $iter \
            --n-enum 100000 \
            --n-accept 500 \
            --mirror-type $mirror \
            --tau-size 0.3 \
            --gamma 0.5 \
            --w1 0 0.125 0.375 0.625 0.75 0.875 1 \
            --w2 1 0.875 0.625 0.375 0.25 0.125 0 \
            --out-dir $res_dir
    done
done

for iter in "${iter_arr[@]}"
do
    for enum in "${enum_arr[@]}"
    do
        for accept in "${accept_arr[@]}"
        do 
            python3 -m vignettes.interference \
                --data-iter $iter \
                --n-enum $enum \
                --n-accept $accept \
                --mirror-type none \
                --tau-size 0.3 \
                --gamma 0.5 \
                --w1 0.25 0.5 0.75 \
                --w2 0.75 0.5 0.25 \
                --out-dir $res_dir
        done
    done
done


# figs
python3 -m vignettes.interference_figs \
    --n-enum 100000 \
    --n-accept 500 \
    --mirror-type all \
    --estimator diff_in_means \
    --tau-size 0.3 \
    --out-dir $res_dir

python3 -m vignettes.interference_figs \
    --n-enum 100000 \
    --n-accept 500 \
    --mirror-type none \
    --estimator diff_in_means \
    --tau-size 0.3 \
    --out-dir $res_dir

python3 -m vignettes.interference_figs \
    --n-enum 100000 \
    --n-accept 500 \
    --mirror-type all \
    --estimator linear_regression \
    --tau-size 0.3 \
    --out-dir $res_dir

python3 -m vignettes.interference_figs \
    --n-enum 100000 \
    --n-accept 500 \
    --mirror-type none \
    --estimator linear_regression \
    --tau-size 0.3 \
    --out-dir $res_dir