import json
import os
from collections import defaultdict
from itertools import chain
from itertools import product
from multiprocessing import Pool
from collections import Counter

from fastcrf.converter import convert_rf_format
from fastcrf.converter import Condition

import pandas as pd
import numpy as np

from research_paper.create_paper_counterfactual_sets_table import get_bounds_query
from research_paper.experiment_manager import ExperimentCVManager

pd.set_option('display.max_columns', 100)

experiments_path = 'research_paper/experiments/'

datasets = [
    'abalone',
    'adult',
    'banknote',
    'compas',
    'credit',
    'mammographic_mases',
    'occupancy',
    'pima',
    'postoperative',
    'seismic'
]

cvs = ['cv_' + str(i) for i in range(10)]
BASE_PATH = "research_paper/experiments/"


def get_rule_rf(rf, sample_data, parsed_rf, prob_threshold):
    all_conditions = []
    foil_idx = sample_data['counterfactual_class']

    all_condition_rules = []

    for rule_id in sample_data['set_rules']:
        _, rule = parsed_rf.rules[rule_id]
        rule_count = rule.label.reshape(-1)
        rule_prob = rule_count / rule.label.sum()
        all_condition_rules.append((rule.conditions, (rule_prob[foil_idx], rule_count[foil_idx])))

    all_condition_rules = sorted(all_condition_rules, key=lambda r: r[1], reverse=True)
    cutoff_prob_cum = len(rf.estimators_) * prob_threshold
    current_prob_cum = 0
    i = 0
    while i < len(all_condition_rules) and current_prob_cum < cutoff_prob_cum:
        rule_conditions, (rule_prob, label_count) = all_condition_rules[i]
        all_conditions.extend(rule_conditions)
        current_prob_cum += rule_prob
        i += 1

    simplified_conditions = dict()
    features = set()
    for feature, threshold, is_leq in all_conditions:
        key = (feature, is_leq)
        features.add(feature)
        if key not in simplified_conditions:
            simplified_conditions[key] = threshold
        else:
            previous_threshold = simplified_conditions[key]

            if (is_leq and threshold < previous_threshold) or (not is_leq and threshold > previous_threshold):
                simplified_conditions[key] = threshold

    rule = dict()
    for feature in features:
        lt = simplified_conditions.get((feature, True), None)
        gt = simplified_conditions.get((feature, False), None)

        rule[feature] = (gt, lt)
    return rule


def process_dataset_threshold_cv(experiment_cv_manager, threshold):
    X_train = experiment_cv_manager.X_train
    rf = experiment_cv_manager.rf
    dataset_info = experiment_cv_manager.datataset_info
    base_path = os.path.join(experiment_cv_manager.experiment_path, 'rf_ocse')
    samples = []
    X_train.columns = [col.replace('-', '_') for col in experiment_cv_manager.X_train.columns]
    parsed_rf = convert_rf_format(rf, dataset_info)

    if os.path.exists(base_path):
        for sample_file in os.listdir(base_path):
            if sample_file.startswith('sample_') and sample_file.endswith('.json'):
                sample_id = sample_file[:-5]

                with open(os.path.join(base_path, sample_file)) as fin:
                    sample_data = json.load(fin)

                prop_counterfactual = None
                covered_samples_class = []
                bounds = get_rule_rf(rf, sample_data, parsed_rf, threshold)
                bounds = {dataset_info.dataset_description[int(k)]['name']: v for k, v in bounds.items()}

                query = get_bounds_query(bounds)
                if len(query) > 0:
                    covered = X_train.query(query)
                    if len(covered) > 0:
                        covered_samples_class = experiment_cv_manager.rf.predict(covered.values)
                        counts = Counter(covered_samples_class)
                        foil_class = sample_data['counterfactual_class']

                        if foil_class in counts:
                            prop_counterfactual = counts[foil_class] / len(covered_samples_class)
                        else:
                            prop_counterfactual = 0

                sample_res = {
                    'cv': experiment_cv_manager.cv,
                    'sample_id': sample_id,
                    'dataset': experiment_cv_manager.dataset_name,
                    ('prop_populated', "th_" + str(threshold)): min(1, len(covered_samples_class)),
                    ('prop_covered_samples', "th_" + str(threshold)): len(covered_samples_class) / len(X_train),
                    ('prop_counterfactual', "th_" + str(threshold)): prop_counterfactual
                }

                samples.append(sample_res)

    return samples


def process_dataset_threshold(args):
    dataset, cv, threshold = args
    methods_samples = defaultdict(dict)
    cv_manager = ExperimentCVManager.read_experiment(os.path.join(BASE_PATH, dataset, cv))

    for sample in process_dataset_threshold_cv(cv_manager, threshold):
        key = (sample['cv'], sample['sample_id'], sample['dataset'])
        methods_samples[key].update(sample)

    return list(methods_samples.values())


def generate_df(datasets, cvs, thresholds):
    args = list(product(datasets, cvs, thresholds))

    with Pool() as pool:
        samples = pool.map(process_dataset_threshold, args)

    return list(chain(*samples))


threshoolds = [0.4, 0.3, 0.2, 0.1]
result_df = pd.DataFrame(generate_df(datasets, cvs, threshoolds))
result_df.set_index(['cv', 'sample_id', 'dataset'], inplace=True)
result_df.columns = pd.MultiIndex.from_tuples([(col,) if isinstance(col, str) else col for col in result_df.columns])
result_df.sort_index(axis=1, level=[0, 1], inplace=True)

dynamic_data = []
for index, row in result_df.iterrows():

    prop_counterfactuals = None
    prop_covered_samples = None
    prop_populated = None

    for threshold in threshoolds:
        name = 'th_' + str(threshold)

        if row['prop_covered_samples', name] > 0:
            prop_counterfactuals = row['prop_counterfactual', name]
            prop_covered_samples = row['prop_covered_samples', name]
            prop_populated = row['prop_populated', name]
            break

    dynamic_data.append({
        'cv': index[0],
        'sample_id': index[1],
        'dataset': index[2],
        'prop_counterfactual': prop_counterfactuals,
        'prop_covered_samples': prop_covered_samples,
        'prop_populated': prop_populated
    })

dynamic_data_df = pd.DataFrame(dynamic_data)
dynamic_data_df.set_index(['cv', 'sample_id', 'dataset'], inplace=True)
dynamic_data_df.columns = pd.MultiIndex.from_tuples([(col, 'rf_ocse_dynamic')  for col in dynamic_data_df.columns])
dynamic_data_df.sort_index(axis=1, level=[0, 1], inplace=True)

df_result_all = result_df.join(dynamic_data_df)
df_result_all_mean = (df_result_all * 100).groupby(level=2).mean().round(2)

df_report = pd.DataFrame()

discard_thresholds = ['0.1', '0.3']

for method in set(df_result_all.columns.get_level_values(1)):
    if not any(x in method for x in discard_thresholds):
        df_report[method, 's-fidelity'] = df_result_all_mean['prop_counterfactual', method]
        df_report[method, 'set coverage'] = df_result_all_mean['prop_covered_samples', method].astype(str) + \
                                         " (" + df_result_all_mean['prop_populated', method].astype(str) + "%)"

df_report.columns = pd.MultiIndex.from_tuples(df_report.columns)

df_report.sort_index(level=1, inplace=True)
df_report.to_csv('research_paper/reports/pcs_table.csv')


def store_latex(df, file):
    with open('research_paper/reports/' + file, 'w') as fout:
        fout.write(df.to_latex())


store_latex(df_report, 'pcs_table.latex')