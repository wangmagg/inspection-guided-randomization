#!/bin/bash

declare -a enum_arr=(10000 30000 50000 750000 100000 150000) 
declare -a accept_arr=(100 500 1000)  
declare -a tau_size_arr=(0.3 0)
declare -a mirror_arr=("all" "good")
declare -a gamma_arr=(0.25 0.75)

for enum in "${enum_arr[@]}"
do
    for accept in "${accept_arr[@]}"
    do
        python3 -m vignettes.interference \
            --n-enum $enum \
            --n-accept $accept \
            --mirror-type all \
            --tau-size 0.3 \
            --gamma 0.5 \
            --n-data-samples 5 \
            --w1 0.5 \
            --w2 0.5
    done
done

for gamma in "${gamma_arr[@]}"
do
    python3 -m vignettes.interference \
        --n-enum 100000 \
        --n-accept 500 \
        --mirror-type all \
        --gamma $gamma \
        --w1 0.5 \
        --w2 0.5 \
        --n-data-samples 5
done

for tau_size in "${tau_size_arr[@]}"
do
    python3 -m vignettes.interference \
        --n-enum 100000 \
        --n-accept 500 \
        --mirror-type all \
        --tau-size $tau_size \
        --gamma 0.5 \
        --w1 0.25 0.5 0.75 0 0.125 0.375 0.625 0.875 1 \
        --w2 0.75 0.5 0.25 1 0.875 0.625 0.375 0.125 0 \
        --n-data-samples 5
done

for mirror in "${mirror_arr[@]}"
do
    python3 -m vignettes.interference \
        --n-enum 100000 \
        --n-accept 500 \
        --mirror-type $mirror \
        --tau-size 0.3 \
        --gamma 0.5 \
        --w1 0.25 0.5 0.75 0 0.125 0.375 0.625 0.875 1 \
        --w2 0.75 0.5 0.25 1 0.875 0.625 0.375 0.125 0 \
        --n-data-samples 5
done

# figs
python3 -m vignettes.interference --n-enum 100000 --n-accept 500 --mirror-type all --tau-size 0.3