# Criteria-based Randomization: A Transparent and Flexible Restricted Randomization Framework for Better Experimental Design
When designing a randomized experiment, some treatment allocations are better than others for answering the causal question we want to answer, i.e. will lead to less biased, more precise treatment effect estimates. For example, allocations that are balanced on outcome-prognostic covariates will increase the precision of a difference-in-means estimator. Restricted randomization can be used to filter out undesirable treatment allocations that will likely generate poor effect estimates; that is, by restricting the randomization, undesirable allocations cannot possibly be chosen as the official deployed allocation for the experiment. Traditionally, restricted randomization is applied to two-arm experiments and focuses on covariate balance as the metric by which to assess the desirability of a candidate treatment allocation. 

**Criteria-based Randomization (CBR)** is a restricted randomization framework that lets us flexibly _define_ and _apply_ domain-driven design criteria beyond two-arm covariate balance. Importantly, CBR is transparent and reproducible. During the design of the experiment, the space of acceptable randomization allocations and the steps by which it is derived is “locked in” ex ante and can be pre-registered, safeguarding against dubious statistical “hacking” in the analysis and inference phase of the experiment. We break CBR down into the following steps: </br>
  1. _Specification_ of the design criteria </br>
  2. _Enumeration_ of a candidate pool of allocations </br>
  3. _Evaluation_ of each allocation against the design criteria </br>
  4. _Restriction_ of the pool of allocations to those that meet the criteria </br>
  5. _Pre-registration_ of the design criteria and the accepted pool of allocations </br>
  6. _Randomization_ of the allocation drawn for the official experiment </br>

This repository contains simulation code for three illustrative vignettes that highlight the utility of CBR in distinct experimental settings: multi-arm experiments, group formation experiments, and experiments with interference.

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
    .
    ├── params                  # Files with values for fixed input parameters
    ├── scripts                 # Bash scripts for running simulations
    ├── src                     # Source files
      ├── analysis              # Functions for estimation and inference
      ├── design                # Functions for experimental designs
      ├── models                # Functions for data generation
      ├── sims                  # Simulations of three illustrative CBR use cases
      ├── visualization         # Functions to plot results from simulations
    └── README.md

## Usage
### Running Vignette Simulations
#### 1. Multi-arm Experiments ####

In a multi-arm experiment, we simultaneously evaluate multiple interventions and make pairwise comparisons between arms to estimate treatment effects. A well-designed experiment in this setting has covariates balanced across all pairs of arms that will be compared in the analysis phase of the experiment. To use CBR, we construct and apply multi-arm balance criteria.

Our multi-arm experiment simulation is modeled based on an educational intervention study conducted to test how different types of feedback on class assignments affect students’ grades. We examine the allocations that are accepted under a CBR design scheme and compare them with allocations accepted under a naive complete randomization design and under a threshold blocking design (Higgens et al, 2016). 

For convenience, we provide bash scripts that run the experiment for multiple data replicates across several different parameter settings (number of arms, number of individuals per arm, number of candidate allocations generated): <br />
```
bash scripts/run_multarm_sims.sh --make-data
bash scripts/run_multarm_sims.sh --run-trial
```

Running these scripts creates: </br>
  1. ```output/mult-arm``` directory that stores a pickled object for each simulated experiment
  2. ```res/mult-arm``` directory that stores CSV result files

#### 2. Group Formation Experiments ####

In a group formation experiment, we randomize individuals into different groups of certain compositions. For example, a group may be the set of college freshmen assigned to the same dorm room, and composition may be the number of individuals in a room who were “high academic achievers” in high school. As in the multi-arm setting, we would like to make pairwise comparisons between groups of different compositions. A well-designed experiment in this setting has groups balanced on all observed covariates other than salient attributes that are used to define composition. To use CBR, we can apply multi-arm balance criteria and exclude the composition-related attributes. 

Our group formation experiment simulation is inspired by stereotype threat interventions that have shown heterogenous effects based on the gender composition of a group, where the magnitude and direction effect
of the intervention can depend on the male-to-female ratio. We use a similar data-generating process to the one for the simulated multi-arm experiment.

For convenience, we provide bash scripts that run the experiment for multiple data replicates across several different parameter settings (number of individuals per arm, number of candidate allocations generated): <br />
```
bash scripts/run_composition_sims.sh --make-data
bash scripts/run_composition_sims.sh --run-trial
```

Running these scripts creates: 

1. ```output/composition```: directory that stores a pickled object for each simulated experiment
2. ```res/composition```: directory that stores CSV result files.

#### 3. Experiments with Network Interference #

Interference occurs when the treatment assigned to one individual affects the outcome of a different individual. A common target estimand in experiments with interference is the total treatment effect, or the difference in outcomes if everyone were to receive treatment compared to if everyone were to receive control.  Failing to properly account for interference can make estimates of the total treatment effect both biased and imprecise. A well-designed experiment in this setting controls spillover while also maintaining inter-cluster balance on covariates. Within the CBR design framework, by leveraging prior knowledge, we can apply interference-related criteria that control the amount of anticipated interference in a more targeted, precise way than with cluster randomization. We can also combine interference-related criteria with balance criteria to control interference while also encouraging covariate balance.

Our simulated experiment with interference is based on a trial of a sexual assault prevention intervention for adolescent girls conducted across schools in urban settlements in Kenya. During the intervention, the girls
learned and practiced verbal and physical skills for resisting sexual assault. The trial was instead randomized on the school level, and, because the settlements were densely populated, it is likely that girls attending different schools interacted with one another. As a result, the skills learned by girls assigned to receive the intervention may have been passed on to girls assigned to the control arm.

For convenience, we provide bash scripts that run the experiment for multiple data replicates across several different parameter settings (number of individuals per arm, number of candidate allocations generated): <br />
```
bash scripts/run_kenya_sims.sh --make-data
bash scripts/run_kenya_sims.sh --run-trial
```
Running these scripts creates: 

1. ```output/composition```: directory that stores a pickled object for each simulated experiment
2. ```res/composition```: directory that stores CSV result files.

### Visualizations

To generate plots of the simulation results, run the following bash scripts:
```
bash scripts/run_multarm_visualization.sh
bash scripts/run_composition_visualization.sh
bash scripts/run_kenya_visualization.sh
```


