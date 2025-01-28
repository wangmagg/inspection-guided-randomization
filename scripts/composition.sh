#!/bin/bash
iter_arr=(0 1 2 3 4)
res_dir="res_resub/composition"

for iter in "${iter_arr[@]}"
do
    python3 -m vignettes.composition \
        --n-stu 120 \
        --rhos 0.5 0.3 0.7 \
        --n-enum 100000 \
        --n-accept 500 \
        --data-iter $iter \
        --out-dir $res_dir
done

for iter in "${iter_arr[@]}"
do
    python3 -m vignettes.composition \
        --n-stu 120 \
        --rhos 0.5 0.4 0.6 \
        --n-enum 100000 \
        --n-accept 500 \
        --data-iter $iter \
        --out-dir $res_dir
    python3 -m vignettes.composition \
        --n-stu 120 \
        --rhos 0.5 0.2 0.8 \
        --n-enum 100000 \
        --n-accept 500 \
        --data-iter $iter \
        --out-dir $res_dir
done


python3 -m vignettes.composition_figs \
    --rhos 0.5 0.3 0.7 \
    --n-enum 100000 \
    --n-accept 500 \
    --data-iter 0 \
    --out-dir $res_dir