from .run_mace import run_experiment_mace


def run_experiment_mo(experiment_method_manager):
    run_experiment_mace(experiment_method_manager, method='mo')