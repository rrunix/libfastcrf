from .run_mace import run_experiment_mace


def run_experiment_ft(experiment_method_manager):
    run_experiment_mace(experiment_method_manager, method='ft')