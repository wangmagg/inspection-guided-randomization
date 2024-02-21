# Criteria-based Randomization
Code to run simulations for Criteria-based Randomization (CBR), a restricted randomization framework that permits flexibly defined design criteria.

## Setup
__Installation__ <br />
To clone this repository, run the following <br />
```
git clone https://github.com/wangmagg/cbr.git
cd cbr
```

__Python dependencies__ <br />
Python dependencies are specified in `env/requirements.txt`. To set up and activate a Conda environment with the requisite dependicies, run the following <br />
```
conda env create -f env/environment.yml
conda activate cbr-env
```

## File Structure

## Usage
To reproduce the simulation results from the report, run the following scripts (in this order): <br />
```
bash scripts/run_net_simulations.sh
bash scripts/run_outcome_simulations.sh
bash scripts/run_trial_simulations.sh
```
This will create a `output` directory and will store output files as `output / <[NETWORK_MODEL_NAME]> / <n-[SAMPLE SIZE]_it-[TRIAL REPETITIONS]_tau-[TAU]/[RANDOMIZATION DESIGN].pkl>`.

To collect the `.pkl` files across randomization designs into csv files, run the following: <br />
```
python3 -m collect_results --[optional arguments]
```
This will create a `csvs` directory and will store files as `csvs / <[NETWORK_MODEL_NAME]> / <n-[SAMPLE SIZE]_it-[TRIAL REPETITIONS]_tau-[TAU].csv>`

To create the scatterplots and density plots, run the following: <br />
```
bash scripts/make_plots.shh
```
This will create a `figs` directory and will store files as `figs / <[NETWORK_MODEL_NAME]> / <n-[SAMPLE SIZE]_it-[TRIAL REPETITIONS]_tau-[TAU]> / <[NETWORK_MODEL_NAME]_[EXPOSURE_MODEL_NAME].png>`.
