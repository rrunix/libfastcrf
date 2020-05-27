import glob
import itertools
import os
import sys
import warnings

from research_paper.multiprocessing_utils import multiprocessing_context
from research_paper.experiment_manager import ExperimentCVManager
from research_paper.multiprocessing_utils import MultiprocessTQDM


def run(params):
    warnings.simplefilter("ignore")
    method_params, experiment_def = params
    method_name = method_params.pop('name')

    experiment_cv_manager = ExperimentCVManager.read_experiment(experiment_def)
    experiment_manager = experiment_cv_manager.create_method_logger(method_name, method_params)

    if method_name == 'mace':
        from research_paper.methods.run_mace import run_experiment_mace
        run_experiment_mace(experiment_manager)

    elif method_name == 'mo':
        from research_paper.methods.run_mo import run_experiment_mo
        run_experiment_mo(experiment_manager)

    elif method_name == 'ft':
        from research_paper.methods.run_ft import run_experiment_ft
        run_experiment_ft(experiment_manager)

    elif method_name == 'lore':
        from research_paper.methods.run_lore import run_experiment_lore
        run_experiment_lore(experiment_manager)

    elif method_name == 'rf_ocse':
        from research_paper.methods.run_rf_ocse import run_experiment_rf_ocse
        run_experiment_rf_ocse(experiment_manager)

    elif method_name == 'fbt':
        from research_paper.methods.run_fbt import run_experiment_fbt
        run_experiment_fbt(experiment_manager)

    elif method_name == 'defrag':
        from research_paper.methods.run_defrag import run_experiment_defrag
        run_experiment_defrag(experiment_manager)
    else:
        raise ValueError("Method {name} not implemented".format(name=method_name))


include = [
    # 'credit',
    # 'pima',
    'pima',
    # 'occupancy',
    # 'compas',
    # 'adult',
    # 'mammographic_mases',
    # 'ionosphere',
    # 'postoperative',
    # 'banknote',
    # 'compas',
    # 'adult',
    # 'cv_0'
]

exclude = [
    # 'ionosphere',
    # 'adult',
    # #'pima',
    # 'wine'
]


methods = [
    {'name': 'rf_ocse', 'overwrite': False}
]


def should_include(experiment):
    if len(include) > 0 and not any(x in experiment for x in include):
        return False

    if len(exclude) > 0 and any(x in experiment for x in exclude):
        return False

    return True


experiments = glob.glob('research_paper/experiments/*/*/')
filtered_experiments = [experiment for experiment in experiments if should_include(experiment)]
# filtered_experiments = ['research_paper/experiments/abalone/cv_0/']
methods_experiments = list(itertools.product(methods, filtered_experiments))


with multiprocessing_context(processes=6) as pool:
    with MultiprocessTQDM(len(methods_experiments), desc='Global progress') as mtqdm:
        for _ in pool.imap(run, methods_experiments):
            mtqdm.update()

