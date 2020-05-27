import numpy as np
import sys
from research_paper.methods.mace import generateFTExplanations
from research_paper.methods.mace import generateMACEExplanations
from research_paper.methods.mace import generateMOExplanations
from research_paper.methods.mace_dataset import convert_to_mace_dataset
from research_paper.multiprocessing_utils import MultiprocessTQDM
from research_paper.methods.mace import normalizedDistance


def generateExplanations(
        method,
        explanation_file_name,
        model_trained,
        dataset_obj,
        factual_sample,
        norm_type_string,
        potential_observable_samples,
        standard_deviations):
    method = method.upper()

    if 'MACE' == method:  # 'MACE_counterfactual':
        epsilon = float('10e-5')
        explanation = generateMACEExplanations.genExp(
            explanation_file_name,
            model_trained,
            dataset_obj,
            factual_sample.to_dict(),
            norm_type_string,
            epsilon
        )

        explanation.pop('all_counterfactuals')
        return explanation

    elif method == 'MO':  # 'minimum_observable':

        return generateMOExplanations.genExp(
            explanation_file_name,
            model_trained,
            dataset_obj,
            factual_sample,
            potential_observable_samples,
            norm_type_string
        )

    elif method == 'FT':  # 'feature_tweaking':

        possible_labels = [0, 1]
        desired_label = (model_trained.predict([factual_sample])[0] + 1) % 2
        epsilon = .5
        return generateFTExplanations.genExp(
            model_trained,
            factual_sample.to_dict(),
            possible_labels,
            desired_label,
            epsilon,
            norm_type_string,
            dataset_obj,
            standard_deviations,
            True
        )
    else:
        raise Exception('{method} not recognized as a valid `approach_string`.'.format(method=method))


def update_names(obs, mapping):
    return {mapping[key]: value for key, value in obs.items()}


def run_experiment_mace(experiment_method_manager, method='mace'):
    dataset_info = experiment_method_manager.experiment_manager.datataset_info
    X_train = experiment_method_manager.experiment_manager.X_train
    X_test = experiment_method_manager.experiment_manager.X_test
    y_train = experiment_method_manager.experiment_manager.y_train
    y_test = experiment_method_manager.experiment_manager.y_test
    rf = experiment_method_manager.experiment_manager.rf

    dataset_obj = convert_to_mace_dataset(dataset_info, X_train.copy(), y_train.copy())
    standard_deviations = list(X_train.std())
    X_test = X_test.copy()

    if method == 'mace':
        X_test['y'] = rf.predict(X_test)

    X_test = X_test.rename(columns=dataset_obj.long_kurz_mapping)
    X_train = X_train.rename(columns=dataset_obj.long_kurz_mapping)
    inv_mapping = dict(zip(dataset_obj.long_kurz_mapping.values(), dataset_obj.long_kurz_mapping.keys()))

    input_cols = [col for col in X_test if col != 'y']

    with MultiprocessTQDM(len(X_test), experiment_method_manager) as mtqdm:
        for idx, factual_obs in X_test.iterrows():
            if not experiment_method_manager.has_been_processed(idx) or experiment_method_manager.overwrite:
                try:
                    res = run_observation(method, rf, X_train, factual_obs, dataset_obj, standard_deviations,
                                          input_cols, experiment_method_manager.log_stdout)

                    res['factual_sample'] = update_names(res['factual_sample'], inv_mapping)
                    res['counterfactual_sample'] = update_names(res['counterfactual_sample'], inv_mapping)
                    experiment_method_manager.log_observation_result(idx, res)
                except Exception as e:
                    experiment_method_manager.log_observation_error(idx, e)

            mtqdm.update()

    experiment_method_manager.finish_experiment()


def run_observation(method, rf, X_train, factual_obs, dataset_obj, standard_deviations, input_cols, log_file):
    explanation_file = log_file
    potential_observable_samples = X_train
    norm_type_string = 'one_norm'
    explanation_object = generateExplanations(
        method,
        explanation_file,
        rf,
        dataset_obj,
        factual_obs,
        norm_type_string,
        potential_observable_samples,  # used solely for minimum_observable method
        standard_deviations,  # used solely for feature_tweaking method,
    )

    cf = explanation_object['counterfactual_sample']

    best_counterfactual_np = np.array([cf[col] for col in input_cols])
    explanation_object['counterfactual_class'] = rf.predict(best_counterfactual_np.reshape(1, -1)).reshape(-1)[0]

    factual_obs_np = np.array([factual_obs[col] for col in input_cols])

    if explanation_object.get('counterfactual_found', True):
        explanation_object['factual_class'] = rf.predict(factual_obs_np.reshape(1, -1)).reshape(-1)[0]
        explanation_object['valid_'] = explanation_object['counterfactual_class'] != explanation_object['factual_class']
    else:
        explanation_object['factual_class'] = -1
        explanation_object['valid_'] = False

    return explanation_object
