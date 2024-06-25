#!/bin/bash

declare -a iter_arr=(0 1 2 3 4)
declare -a enum_arr=(5000 10000 25000 50000 100000 200000) 
declare -a accept_arr=(100 500 1000)  

for iter in "${iter_arr[@]}"
do
    for enum in "${enum_arr[@]}"
    do
        for accept in "${accept_arr[@]}"
        do
            python3 -m vignettes.multarm --n-stu 80 --n-arms 4 --n-enum $enum --n-accept $accept --data-iter $iter
        done
    done
done

# 6 arms
for iter in "${iter_arr[@]}"
do
    for enum in "${enum_arr[@]}"
    do
        for accept in "${accept_arr[@]}"
        do
            python3 -m vignettes.multarm --n-stu 120 --n-arms 6 --tau-sizes 0 0.15 0.3 0.45 0.6 --n-enum 100000 --n-accept 500 --data-iter $iter
        done
    done
done

# Make figures for n-arms:4
python3 -m vignettes.multarm_figs --n-stu 80 --n-arms 4 --n-enum 100000 --n-accept 500 --data-iter 0