from __future__ import print_function

import os
import time

import numpy as np

from research_paper.methods.LORE.lore import explain
from research_paper.methods.LORE.neighbor_generator import genetic_neighborhood
from research_paper.methods.LORE.pyyadt import apply_counterfactual
from research_paper.methods.LORE.util import set_discrete_continuous
from research_paper.methods.mace.normalizedDistance import getDistanceBetweenSamples
from research_paper.methods.mace_dataset import convert_to_mace_dataset
from research_paper.multiprocessing_utils import MultiprocessTQDM
from research_paper.multiprocessing_utils import get_multiprocessing_id
from research_paper.multiprocessing_utils import get_desc

DISTANCE_TYPE = 'one_norm'


def to_normal_str(string_list):
    return [s.encode("utf-8") for s in string_list]


def recognize_features_type(dataset_info):

    type_features = {
        'integer': [],
        'double': to_normal_str(dataset_info.get_feature_type_features(1) +
                  dataset_info.get_feature_type_features(2) + dataset_info.get_feature_type_features(3)),
        'string': to_normal_str(dataset_info.get_feature_type_features(4) + ['y']),
        'integer_based': to_normal_str(dataset_info.get_feature_type_features(2) + dataset_info.get_feature_type_features(3))
    }

    features_type = dict()
    for col in type_features['integer']:
        features_type[col] = 'integer'
    for col in type_features['double']:
        features_type[col] = 'double'
    for col in type_features['string']:
        features_type[col] = 'string'

    return type_features, features_type


def prepare_dataset(dataset_name, X, y, dataset_info):
    X = X.copy()
    input_cols = X.columns

    class_name = dataset_info.class_name
    X.loc[y.index, class_name] = y
    df = X

    # Features Categorization
    columns = X.columns
    possible_outcomes = list(df[class_name].unique())

    type_features, features_type = recognize_features_type(dataset_info)

    discrete = []
    discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete, continuous=None)

    columns_tmp = list(columns)
    columns_tmp.remove(class_name)
    idx_features = {i: col for i, col in enumerate(columns_tmp)}

    # Dataset Preparation for Scikit Alorithms
    # df_le, label_encoder = label_encode(df, discrete)
    X = df.loc[:, df.columns != class_name].values
    y = df[class_name].values

    dataset = {
        'name': dataset_name,
        'df': df,
        'input_cols': input_cols,
        'columns': list(columns),
        'class_name': class_name,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete,
        'continuous': continuous,
        'idx_features': idx_features,
        'label_encoder': None,
        'X': X,
        'y': y,
    }

    return dataset


def adapt_sample_mace(sample, dataset_obj):
    return {kurz: sample[long] for long, kurz in dataset_obj.long_kurz_mapping.items()}


def run_observation(rf, X_train, factual_obs, dataset_lore, dataset_obj, tmp=''):
    start_time = time.time()

    x2e = np.vstack((factual_obs.values.reshape(1, -1), X_train))
    explanation = explain(0, x2e, dataset_lore, rf,
                          ng_function=genetic_neighborhood,
                          discrete_use_probabilities=True,
                          continuous_function_estimation=False,
                          returns_infos=False,
                          path=tmp, sep=';', log=False)

    rule, counterfactuals = explanation

    valid = False
    best_distance = None
    best_counterfactual = None
    best_rule = None

    factual_obs_dict = factual_obs.to_dict()
    factual_obs_dict['y'] = rf.predict([factual_obs])[0]

    for counterfactual in counterfactuals:
        cc = apply_counterfactual(factual_obs_dict, counterfactual, dataset_lore['continuous'], dataset_lore['discrete'],
                                  dataset_lore['features_type'], dataset_lore['type_features']['integer_based'])

        cc_np = np.array([cc[col] for col in dataset_lore['input_cols']])
        cc['y'] = rf.predict([cc_np])[0]

        distance = getDistanceBetweenSamples(adapt_sample_mace(factual_obs_dict, dataset_obj),
                                             adapt_sample_mace(cc, dataset_obj), DISTANCE_TYPE, dataset_obj)

        if best_distance is None or best_distance > distance:
            best_distance = distance
            best_counterfactual = cc
            valid = cc['y'] != factual_obs_dict['y']
            best_rule = counterfactual

    end_time = time.time()

    return {
        'factual_sample': factual_obs_dict,
        'factual_class': factual_obs_dict['y'],
        'counterfactual_class': best_counterfactual['y'],
        'counterfactual_sample': best_counterfactual,
        'counterfactual_found': valid,
        'counterfactual_rule': best_rule,
        'counterfactual_distance': best_distance if best_counterfactual is not None else -1,
        'counterfactual_time': end_time - start_time,
    }


def run_experiment_lore(experiment_method_manager):
    dataset_info = experiment_method_manager.experiment_manager.datataset_info
    X_train = experiment_method_manager.experiment_manager.X_train
    X_test = experiment_method_manager.experiment_manager.X_test
    y_train = experiment_method_manager.experiment_manager.y_train
    rf = experiment_method_manager.experiment_manager.rf
    dataset_lore = prepare_dataset('files_' + str(get_multiprocessing_id()) , X_train, y_train, dataset_info)
    dataset_obj_mace = convert_to_mace_dataset(dataset_info, X_train.copy(), y_train)

    tmp_folder = 'research_paper/tmp/'
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    X_train = X_train.values

    with MultiprocessTQDM(len(X_test), experiment_method_manager) as mtqdm:
        for idx, factual_obs in X_test.iterrows():
            try:
                res = run_observation(rf, X_train, factual_obs, dataset_lore, dataset_obj_mace, tmp=tmp_folder)
                experiment_method_manager.log_observation_result(idx, res)
            except Exception as e:
                experiment_method_manager.log_observation_error(idx, e)

            mtqdm.update()

    experiment_method_manager.finish_experiment()
