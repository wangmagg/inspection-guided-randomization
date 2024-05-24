import pickle
from pathlib import Path
from typing import Dict, Tuple, List

from src.design.fitness_functions import Fitness
from src.sims.trial import SimulatedTrial
from src.sims.run_kenya_trial import SimulatedKenyaTrial
from src.sims.run_multarm_trial import SimulatedMultiArmTrial
from src.sims.run_network_trial import SimulatedNetworkTrial


def multarm_trial_set(
    trial_path:Path,
) -> Tuple[Dict[str, SimulatedMultiArmTrial], Fitness]:
    trial_set = {}

    n_z_subdir = trial_path.parent.parent
    n_per_arm_subdir = n_z_subdir.parent

    # Load complete randomization results
    cr_trial_fname = n_z_subdir / "rand-complete" / trial_path.name
    if cr_trial_fname.exists():
        with open(cr_trial_fname, "rb") as f:
            cr_trial = pickle.load(f)
        cr_trial.mapping, cr_trial.use_cols = None, None
        cr_trial.set_data_from_config()
        cr_trial.set_design_from_config()
        trial_set[cr_trial.rand_mdl.plotting_name] = cr_trial

    # Load quickblock randomization results
    qb_trial_fname = n_per_arm_subdir / "rand-quick-block" / trial_path.name
    if qb_trial_fname.exists():
        with open(qb_trial_fname, "rb") as f:
            qb_trial = pickle.load(f)
        qb_trial.mapping, qb_trial.use_cols = None, None
        qb_trial.set_data_from_config()
        qb_trial.set_design_from_config()
        trial_set[qb_trial.rand_mdl.plotting_name] = qb_trial

    # Load restricted randomization results
    with open(trial_path, "rb") as f:
        trial = pickle.load(f)
    trial.mapping, trial.use_cols = None, None
    trial.set_data_from_config()
    trial.set_design_from_config()
    trial_set[trial.rand_mdl.plotting_name] = trial

    # Load restricted randomization with genetic search results
    genetic_trial_fname = (
        n_z_subdir
        / f"rand-restricted-genetic_{trial.rand_mdl.fitness_fn.name}"
        / trial_path.name
    )

    if genetic_trial_fname.exists():
        with open(genetic_trial_fname, "rb") as f:
            genetic_trial = pickle.load(f)
        genetic_trial.mapping, genetic_trial.use_cols = None, None
        genetic_trial.set_data_from_config()
        genetic_trial.set_design_from_config()
        trial_set[genetic_trial.rand_mdl.plotting_name] = genetic_trial

    return trial_set, trial.rand_mdl.fitness_fn

def composition_trial_set(
    trial_path: Path
) -> Tuple[Dict[str, SimulatedTrial], Fitness]:
    trial_set = {}

    # Load complete randomization results
    cr_trial_fname = trial_path.parent.parent / "rand-group-formation" / trial_path.name

    if cr_trial_fname.exists():
        with open(cr_trial_fname, "rb") as f:
            cr_trial = pickle.load(f)
        cr_trial.set_data_from_config()
        cr_trial.mapping = None
        cr_trial.set_design_from_config()
        trial_set[cr_trial.rand_mdl.plotting_name] = cr_trial

    # Load restricted randomization results
    with open(trial_path, "rb") as f:
        trial = pickle.load(f)
    trial.set_data_from_config()
    trial.mapping = None
    trial.set_design_from_config()
    trial_set[trial.rand_mdl.plotting_name] = trial

    # Load restricted randomization with genetic search results
    genetic_trial_fname = (
        trial_path.parent.parent
        / f"rand-group-formation-restricted-genetic_{trial.rand_mdl.fitness_fn.name}"
        / trial_path.name
    )

    if genetic_trial_fname.exists():
        with open(genetic_trial_fname, "rb") as f:
            genetic_trial = pickle.load(f)
        genetic_trial.set_data_from_config()
        genetic_trial.mapping = None
        genetic_trial.set_design_from_config()
        trial_set[genetic_trial.rand_mdl.plotting_name] = genetic_trial

    return trial_set, trial.rand_mdl.fitness_fn

def kenya_trial_set(
    trial_path: Path, 
    include_settlement_cluster: bool = False
) -> Tuple[Dict[str, SimulatedKenyaTrial], Fitness]:
    
    trial_set = {}

    # Load complete randomization results for school-level clustering
    cr_trial_fname = trial_path.parent.parent / "rand-complete_cluster-school" / trial_path.name
    with open(cr_trial_fname, "rb") as f:
        cr_trial = pickle.load(f)
    cr_trial.set_data_from_config()
    cr_trial.set_design_from_config()
    trial_set[f"{cr_trial.rand_mdl.plotting_name} (Sch)"] = cr_trial

    # Load complete randomization results for settlement-level clustering
    if include_settlement_cluster:
        cr_set_trial_fname = (
            trial_path.parent.parent  / "rand-complete_cluster-settlement" / trial_path.name
        )
        with open(cr_set_trial_fname, "rb") as f:
            cr_set_trial = pickle.load(f)
        cr_set_trial.set_data_from_config()
        cr_set_trial.set_design_from_config()
        trial_set[f"{cr_set_trial.rand_mdl.plotting_name} (Set)"] = cr_set_trial

    # Load restricted randomization results
    with open(trial_path, "rb") as f:
        trial = pickle.load(f)
    trial.set_data_from_config()
    trial.set_design_from_config()
    trial_set[trial.rand_mdl.plotting_name] = trial

    # Load restricted randomization with genetic search results
    genetic_trial_fname = (
        trial_path.parent.parent
        / f"rand-restricted-genetic_{trial.rand_mdl.fitness_fn.name}_cluster-school"
        / trial_path.name
    )

    if genetic_trial_fname.exists():
        with open(genetic_trial_fname, "rb") as f:
            genetic_trial = pickle.load(f)
        genetic_trial.set_data_from_config()
        genetic_trial.set_design_from_config()
        trial_set[genetic_trial.rand_mdl.plotting_name] = genetic_trial
            
    return trial_set, trial.rand_mdl.fitness_fn

def kenya_trial_sets(
    rand_mdl_subdir: Path, 
    include_settlement_cluster: bool = True
) -> Tuple[Dict[int, Dict[str, SimulatedKenyaTrial]], Dict[int, Fitness]]:
    trial_sets = {}
    fitness_fns = {}
    for trial_fname in rand_mdl_subdir.glob("*.pkl"):
        print(trial_fname)
        trial_set = {}

        # Load complete randomization results for school-level clustering
        cr_trial_fname = rand_mdl_subdir.parent / "rand-complete_cluster-school" / trial_fname.name
        with open(cr_trial_fname, "rb") as f:
            cr_trial = pickle.load(f)
        cr_trial.set_data_from_config()
        cr_trial.set_design_from_config()
        trial_set[f"{cr_trial.rand_mdl.plotting_name} (Sch)"] = cr_trial

        # Load complete randomization results for settlement-level clustering
        if include_settlement_cluster:
            cr_set_trial_fname = (
                rand_mdl_subdir.parent  / "rand-complete_cluster-settlement" / trial_fname.name
            )
            with open(cr_set_trial_fname, "rb") as f:
                cr_set_trial = pickle.load(f)
            cr_set_trial.set_data_from_config()
            cr_set_trial.set_design_from_config()
            trial_set[f"{cr_set_trial.rand_mdl.plotting_name} (Set)"] = cr_set_trial

        # Load restricted randomization results
        with open(trial_fname, "rb") as f:
            trial = pickle.load(f)
        trial.set_data_from_config()
        trial.set_design_from_config()
        trial_set[trial.rand_mdl.plotting_name] = trial
        fitness_fns[int(trial_fname.stem)] = trial.rand_mdl.fitness_fn

        # Load restricted randomization with genetic search results
        genetic_trial_fname = (
            rand_mdl_subdir.parent
            / f"rand-restricted-genetic_{trial.rand_mdl.fitness_fn.name}_cluster-school"
            / trial_fname.name
        )

        if genetic_trial_fname.exists():
            with open(genetic_trial_fname, "rb") as f:
                genetic_trial = pickle.load(f)
            genetic_trial.set_data_from_config()
            genetic_trial.set_design_from_config()
            trial_set[genetic_trial.rand_mdl.plotting_name] = genetic_trial

        trial_sets[int(trial_fname.stem)] = trial_set

    return trial_sets, fitness_fns
    
def kenya_trial_set_for_hist2d(
        param_subdir_name: str,
        output_dir_name: str,
        exp_subdir_name: str,
        net_mdl_subdir_names: List[str],
        po_mdl_subdir_name: str,
        n_z_subdir_name: str,
        rand_mdl_subdir_name: str
) -> Dict[str, SimulatedKenyaTrial]:
    trial_set = {}
    for net_mdl_subdir_name in net_mdl_subdir_names:
        trial_pkl_dir = (
            Path(output_dir_name)
            / exp_subdir_name
            / param_subdir_name
            / net_mdl_subdir_name
            / po_mdl_subdir_name
            / n_z_subdir_name
            / rand_mdl_subdir_name
        )
        for trial_fname in trial_pkl_dir.glob("*.pkl"):
            with open(trial_fname, "rb") as f:
                trial = pickle.load(f)
            trial.mapping, trial.use_cols = None, None
            trial.set_data_from_config()
            trial.set_design_from_config()

            key = r"$\gamma=$" + str(trial.config.gamma)
            trial_set[key] = trial
            
    return trial_set

def network_trial_set(
    trial_path: Path
) -> Tuple[Dict[str, SimulatedNetworkTrial], Fitness]:

    trial_set = {}

    # Load complete randomization results
    cr_trial_fname = trial_path.parent.parent / "rand-complete" / trial_path.name
    if cr_trial_fname.exists():
        with open(cr_trial_fname, "rb") as f:
            cr_trial = pickle.load(f)
        cr_trial.set_data_from_config()
        cr_trial.set_design_from_config()
        trial_set[cr_trial.rand_mdl.plotting_name] = cr_trial

    # Load graph randomization results
    graph_trial_fname = trial_path.parent.parent / "rand-graph" / trial_path.name
    if graph_trial_fname.exists():
        with open(graph_trial_fname, "rb") as f:
            graph_trial = pickle.load(f)
        mapping = graph_trial.mapping
        graph_trial.set_data_from_config()
        graph_trial.mapping = mapping
        graph_trial.set_design_from_config()
        trial_set[graph_trial.rand_mdl.plotting_name] = graph_trial

    # Load restricted randomization results
    with open(trial_path, "rb") as f:
        trial = pickle.load(f)
    trial.set_data_from_config()
    trial.set_design_from_config()
    trial_set[trial.rand_mdl.plotting_name] = trial

    # Load restricted randomization with genetic search results
    genetic_trial_fname = (
        trial_path.parent.parent
        / f"rand-restricted-genetic_{trial.rand_mdl.fitness_fn.name}"
        / trial_path.name
    )

    if genetic_trial_fname.exists():
        with open(genetic_trial_fname, "rb") as f:
            genetic_trial = pickle.load(f)
        genetic_trial.set_data_from_config()
        genetic_trial.set_design_from_config()
        trial_set[genetic_trial.rand_mdl.plotting_name] = genetic_trial

    return trial_set, trial.rand_mdl.fitness_fn
