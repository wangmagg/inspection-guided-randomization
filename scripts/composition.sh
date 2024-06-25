#!/bin/bash

python3 -m vignettes.composition --n-stu 120 --rhos 0.5 0.3 0.7 --n-enum 100000 --n-accept 500 --n-data-samples 5
python3 -m vignettes.composition --n-stu 120 --rhos 0.5 0.4 0.6 --n-enum 100000 --n-accept 500 --n-data-samples 5
python3 -m vignettes.composition --n-stu 120 --rhos 0.5 0.2 0.8 --n-enum 100000 --n-accept 500 --n-data-samples 5

python3 -m vignettes.composition_figs --rhos 0.5 0.3 0.7 --n-enum 100000 --n-accept 500 --data-iter 0