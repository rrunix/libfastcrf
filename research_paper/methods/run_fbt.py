import bz2
import math
import os
import time
from collections import defaultdict

from research_paper.methods.mace.normalizedDistance import getDistanceBetweenSamples
from research_paper.methods.mace_dataset import convert_to_mace_dataset
from research_paper.multiprocessing_utils import MultiprocessTQDM
from research_paper.methods.forest_based_tree.ConjunctionSet import ConjunctionSet
from research_paper.methods.forest_based_tree.NewModelBuilder import *

max_number_of_branches = 3000


def get_fbt_feature_types(dataset_info):
    return ['float' if attr['type'] == 1 else 'int' for attr in dataset_info.dataset_description.values()]


def get_feature_bounds(conditions):
    bounds = defaultdict(lambda: [None, None])

    for condition in conditions:
        current_bounds = bounds[condition['feature']]

        if condition['is_leq']:
            current_bounds[1] = min(condition['value'], current_bounds[1] or condition['value'])
        else:
            current_bounds[0] = max(condition['value'], current_bounds[0] or condition['value'])

    return dict(bounds)


def extract_rules(fbt_model, training_df):
    def _convert(node, prev_conds):
        if node.left is None and node.right is None:
            return [
                {'feature_conds': get_feature_bounds(prev_conds), 'label': np.argmax(node.node_probas(training_df))}]
        else:
            rules_left = _convert(node.left, prev_conds + [
                {'feature': node.split_feature, 'value': node.split_value, 'is_leq': True}])
            rules_right = _convert(node.right, prev_conds + [
                {'feature': node.split_feature, 'value': node.split_value, 'is_leq': False}])
            return rules_left + rules_right

    return _convert(fbt_model, [])


def index_rules_by_label(rule_set):
    label_rules = defaultdict(list)

    for rule in rule_set:
        label_rules[rule['label']].append(rule)

    return label_rules


def simplify_ensemble(rf, x_train, y_train, feature_types, max_number_of_branches=3000, seed=0):
    np.random.seed(seed)
    # Create the conjunction set
    cs = ConjunctionSet(x_train.columns, x_train.values, x_train.values, y_train.values, rf, feature_types,
                        max_number_of_branches)

    # Train the new models
    branches_df = cs.get_conjunction_set_df().round(decimals=5)
    for i in range(2):
        branches_df[rf.classes_[i]] = [probas[i] for probas in branches_df['probas']]
    df_dict = {}
    for col in branches_df.columns:
        df_dict[col] = branches_df[col].values
    new_model = Node([True] * len(branches_df))
    new_model.split(df_dict)
    return index_rules_by_label(extract_rules(new_model, branches_df))


def run_experiment_fbt(experiment_method_manager):
    dataset_info = experiment_method_manager.experiment_manager.datataset_info
    X_train = experiment_method_manager.experiment_manager.X_train
    X_test = experiment_method_manager.experiment_manager.X_test
    y_train = experiment_method_manager.experiment_manager.y_train
    feature_types = get_fbt_feature_types(dataset_info)
    rf = experiment_method_manager.experiment_manager.rf
    dataset_obj_mace = convert_to_mace_dataset(dataset_info, X_train.copy(), y_train)

    rule_set_file = os.path.join(experiment_method_manager.experiment_method_path, "ruleset.pickle")
    if os.path.exists(rule_set_file):
        with bz2.BZ2File(rule_set_file, 'rb') as fin:
            rule_set_info = pickle.load(fin)
            rule_set = rule_set_info['rule_set']
    else:
        start_time_rule_set = time.time()
        rule_set = simplify_ensemble(rf, X_train, y_train, feature_types)

        rule_set_info = {
            'rule_set': rule_set,
            'extraction_time': time.time() - start_time_rule_set
        }

        with bz2.BZ2File(rule_set_file, 'wb') as fout:
            pickle.dump(rule_set_info, fout)

    base_time = rule_set_info['extraction_time'] / len(X_test)

    with MultiprocessTQDM(len(X_test), experiment_method_manager) as mtqdm:
        for idx, factual_obs in X_test.iterrows():
            try:
                res = run_observation(factual_obs, rf, rule_set, dataset_info, dataset_obj_mace, 0)
                experiment_method_manager.log_observation_result(idx, res)
            except Exception as e:
                experiment_method_manager.log_observation_error(idx, e)

            mtqdm.update()


def satisfy_rule(feature_bounds, factual_obs, dataset_info, epsilon=0.005):
    obs = factual_obs.copy()

    for attr, attr_info in dataset_info.dataset_description.items():
        attr_idx = attr_info['original_position']
        if attr_idx in feature_bounds:
            gt, leq = feature_bounds[attr]

            if attr_info['type'] == 1:
                gt, leq = feature_bounds[attr]

                if gt is not None and (factual_obs[attr_idx] < gt or abs(factual_obs[attr_idx] - gt) < epsilon):
                    obs[attr_idx] = gt + epsilon
                elif leq is not None and (factual_obs[attr_idx] > leq or abs(factual_obs[attr_idx] - leq) < epsilon):
                    obs[attr_idx] = leq - epsilon

            elif attr_info['type'] == 2:
                if gt is not None and (factual_obs[attr_idx] < gt or abs(factual_obs[attr_idx] - gt) < epsilon):
                    obs[attr_idx] = math.floor(gt + 1)
                elif leq is not None and (factual_obs[attr_idx] > leq or abs(factual_obs[attr_idx] - leq) < epsilon):
                    obs[attr_idx] = math.floor(leq)

            elif attr_info['type'] == 3:
                if gt is not None:
                    obs[attr_idx] = 1
                else:
                    obs[attr_idx] = 0
            else:
                raise ValueError("Type not supported")

    return obs


def adapt_sample_mace(sample, dataset_obj):
    obs = {kurz: sample[long] for long, kurz in dataset_obj.long_kurz_mapping.items() if long != 'y'}
    obs['y'] = False
    return obs


def run_observation(factual_obs, rf, rule_set, dataset_info, dataset_obj_mace, base_time):
    factual_class = rf.predict([factual_obs])[0]
    foil_class = (factual_class + 1) % 2
    foil_rules = rule_set[foil_class]

    best_rule = None
    best_obs = None
    best_distance = None
    best_rule_id = None
    start_time = time.time()

    for r_id, foil_rule in enumerate(foil_rules):
        rule_obs = satisfy_rule(foil_rule['feature_conds'], factual_obs, dataset_info)
        obs_distance = getDistanceBetweenSamples(adapt_sample_mace(factual_obs, dataset_obj_mace),
                                             adapt_sample_mace(rule_obs, dataset_obj_mace), 'one_norm', dataset_obj_mace)

        if best_distance is None or best_distance > obs_distance:
            best_distance = obs_distance
            best_rule = foil_rule
            best_obs = rule_obs
            best_rule_id = r_id

    foil_class = rf.predict([best_obs])[0]
    end_time = time.time()

    return {
        'factual_sample': factual_obs.to_dict(),
        'counterfactual_sample': best_obs.to_dict(),
        'factual_class': factual_class,
        'counterfactual_class': foil_class,
        'counterfactual_found': foil_class != factual_class,
        'counterfactual_rule_id': best_rule_id,
        'counterfactual_rule': best_rule['feature_conds'],
        'counterfactual_distance': best_distance if best_rule is not None else -1,
        'counterfactual_time': base_time + (end_time - start_time),
    }
