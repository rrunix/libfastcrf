import bz2
import os
import pickle
import time

from research_paper.experiment_manager import ExperimentCVManager
from research_paper.multiprocessing_utils import MultiprocessTQDM
from research_paper.methods.defragtrees.defragtrees import DefragModel
from research_paper.methods.mace_dataset import convert_to_mace_dataset


def parse_rule(rule_id, mdl):
    bounds =  {feat: [None, None] for feat in range(len(mdl.featurename_))}
    for feat, cmp, value in mdl.rule_[rule_id]:
        feat_bounds = bounds[feat-1]
        if cmp == 1:
            feat_bounds[0] = value if feat_bounds[0] is None else max(value, feat_bounds[0])
        else:
            feat_bounds[1] = value if feat_bounds[1] is None else min(value, feat_bounds[1])

    return {'class': mdl.pred_[rule_id], 'bounds': bounds}


def degrag_ensemble(X_train, y_train, rf, Kmax = 10):
    splitter = DefragModel.parseSLtrees(rf)
    mdl = DefragModel(modeltype='classification', maxitr=100, qitr=0, tol=1e-6, restart=20, verbose=0)
    mdl.fit(X_train.values, y_train.values, splitter, Kmax, fittype='FAB')
    return {'rules': [parse_rule(rule_id, mdl) for rule_id in range(len(mdl.rule_))],
            'otherwise': mdl.pred_default_,
            'mdl': mdl}


def run_experiment_defrag(experiment_method_manager):
    dataset_info = experiment_method_manager.experiment_manager.datataset_info
    X_train = experiment_method_manager.experiment_manager.X_train
    X_test = experiment_method_manager.experiment_manager.X_test
    y_train = experiment_method_manager.experiment_manager.y_train
    rf = experiment_method_manager.experiment_manager.rf
    dataset_obj_mace = convert_to_mace_dataset(dataset_info, X_train.copy(), y_train)

    rule_set_file = os.path.join(experiment_method_manager.experiment_method_path, "ruleset.pickle")
    if os.path.exists(rule_set_file):
        with bz2.BZ2File(rule_set_file, 'rb') as fin:
            rule_set_info = pickle.load(fin)
            rule_set = rule_set_info['rule_set']
    else:
        start_time_rule_set = time.time()
        rule_set = degrag_ensemble(X_train, y_train, rf)

        rule_set_info = {
            'rule_set': rule_set,
            'extraction_time': time.time() - start_time_rule_set
        }

        with bz2.BZ2File(rule_set_file, 'wb') as fout:
            pickle.dump(rule_set_info, fout)

    base_time = rule_set_info['extraction_time'] / len(X_test)

    # with MultiprocessTQDM(len(X_test), experiment_method_manager) as mtqdm:
    #     for idx, factual_obs in X_test.iterrows():
    #         try:
    #             res = run_observation(factual_obs, rf, rule_set, dataset_info, dataset_obj_mace, base_time)
    #             experiment_method_manager.log_observation_result(idx, res)
    #         except Exception as e:
    #             experiment_method_manager.log_observation_error(idx, e)
    #
    #         mtqdm.update()


def run_observation(factual_obs, rf, rule_set, dataset_info, dataset_obj_mace, base_time):
    pass


experiment_def = 'research_paper/experiments/banknote/cv_0/'
experiment_cv_manager = ExperimentCVManager.read_experiment(experiment_def)
experiment_manager = experiment_cv_manager.create_method_logger("degrag", {})
run_experiment_defrag(experiment_manager)