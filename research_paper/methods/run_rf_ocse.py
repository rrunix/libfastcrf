from fastcrf.main import batch_extraction
from research_paper.methods.mace_dataset import convert_to_mace_dataset
from research_paper.methods.mace.normalizedDistance import getDistanceBetweenSamples
from research_paper.multiprocessing_utils import MultiprocessTQDM
import numpy as np


def to_mace_sample(sample, dataset_obj, colnames):
    sample_dict = {col:val for col, val in zip(colnames, sample)}
    sample_dict['y'] = 0
    return {kurz: sample_dict[long] for long, kurz in dataset_obj.long_kurz_mapping.items()}


def run_experiment_rf_ocse(experiment_method_manager):
    X_train = experiment_method_manager.experiment_manager.X_train
    X_test = experiment_method_manager.experiment_manager.X_test
    y_train = experiment_method_manager.experiment_manager.y_train
    dataset_info = experiment_method_manager.experiment_manager.datataset_info
    rf = experiment_method_manager.experiment_manager.rf

    dataset_obj_mace = convert_to_mace_dataset(dataset_info, X_train.copy(), y_train)
    colnames = X_train.columns
    # Abalone cv_0 21

    with MultiprocessTQDM(len(X_test), experiment_method_manager) as mtqdm:
        for idx, res in zip(X_test.index, batch_extraction(rf, dataset_info, X_test.values, max_distance=100, log_every=-1, max_iterations=20_000_000,
                                     export_rules_file=None, dataset=X_train.values)):

            if 'explanation' in res:
                cs = res.pop('explanation')
                res['feature_bounds'] = cs.feat_conditions
                res['feature_bounds_compact'] = res.pop('explanation_compact').feat_conditions
                res['feature_bounds_compact_count'] = res.pop('explanation_compact_count').feat_conditions
                res['counterfactual_sample'] = cs.sample_counterfactual(X_test.loc[idx], epsilon=0.005)
                res['factual_class'] = rf.predict(np.array(X_test.loc[idx]).reshape(1, -1))[0]
                res['counterfactual_distance'] = res['distance']
                res['counterfactual_class'] = rf.predict(np.array(res['counterfactual_sample']).reshape(1, -1))[0]
                res['valid_'] = res['factual_class'] != res['counterfactual_class']

                res['distance'] = getDistanceBetweenSamples(
                    to_mace_sample(cs.sample_counterfactual(X_test.loc[idx], epsilon=0.00005), dataset_obj_mace, colnames),
                    to_mace_sample(X_test.loc[idx], dataset_obj_mace, colnames),
                    'one_norm',
                    dataset_obj_mace
                )

                experiment_method_manager.log_observation_result(idx, res)

            mtqdm.update()
