import json
import os
from itertools import product
from collections import defaultdict
import pandas as pd
from multiprocessing import Pool
from itertools import chain

from research_paper.methods.mace_dataset import convert_to_mace_dataset
from research_paper.experiment_manager import ExperimentCVManager
import numpy as np


experiments_path = 'research_paper/experiments/'
methods = ['rf_ocse', 'fbt', 'lore']

datasets = ['abalone',
            'adult',
            'banknote',
            'compas',
            'credit',
            'mammographic_mases',
            'occupancy',
            'pima',
            'postoperative',
            'seismic']

cvs = ['cv_' + str(i) for i in range(10)]

BASE_PATH = "research_paper/experiments/"


def get_bounds(sample, method, experiment_cv_manager):
    try:
        if method == 'lore':
            bounds = {}
            if isinstance(sample['counterfactual_rule'], dict):
                for feature, condition in sample['counterfactual_rule'].items():
                    if condition.count('<') == 2:
                        gt, _, lt = condition.split('<')
                        lt = lt[1:]
                        gt = float(gt)
                        lt = float(lt)
                    elif '<=' in condition:
                        lt = float(condition.replace('<=', ''))
                        gt = None
                    elif '>' in condition:
                        lt = None
                        gt = float(condition.replace('>', ''))
                    else:
                        lt = float(condition)
                        gt = float(condition)
                    bounds[feature] = (gt, lt)

            return bounds
        else:
            if method == 'fbt':
                bounds = sample['counterfactual_rule']
            elif method == 'rf_ocse':
                bounds = {k: v['values'] for k, v in sample['feature_bounds_compact_count'].items()}
            else:
                raise ValueError("Method {method} not supported".format(method=method))

            dataset_info = experiment_cv_manager.datataset_info
            return {dataset_info.dataset_description[int(k)]['name']: v for k, v in bounds.items()}

    except Exception as e:
        print(e, sample)

    return {}


def get_bounds_query(bounds):
    query_list = []
    for feat, (gt, lq) in bounds.items():
        feat = feat.replace('-', '_')
        if lq is not None:
            feat_query = "( `{f}` <= {v} )".format(v=lq, f=feat)
            query_list.append(feat_query)

        if gt is not None:
            feat_query = "( `{f}` > {v} )".format(v=gt, f=feat)
            query_list.append(feat_query)

    query = " & ".join(query_list)
    return query


def get_covered_samples_class(sample, method, experiment_cv_manager, X_train):
    query_list = []
    bounds = get_bounds(sample, method, experiment_cv_manager)
    query = get_bounds_query(bounds)
    if len(query) == 0:
        return []

    covered = X_train.query(query)
    if len(covered) > 0:
        classes = experiment_cv_manager.rf.predict(covered.values)
    else:
        classes = []

    return pd.Series(data=classes, name='y', index=covered.index)


def process_dataset_method_cv(experiment_cv_manager, method):
    X_train = experiment_cv_manager.X_train
    rf = experiment_cv_manager.rf
    base_path = os.path.join(experiment_cv_manager.experiment_path, method)
    samples = []
    X_train.columns = [col.replace('-', '_') for col in experiment_cv_manager.X_train.columns]

    if os.path.exists(base_path):
        for sample_file in os.listdir(base_path):
            if sample_file.startswith('sample_') and sample_file.endswith('.json'):
                sample_id = sample_file[:-5]

                with open(os.path.join(base_path, sample_file)) as fin:
                    sample_data = json.load(fin)

                covered_samples_class = get_covered_samples_class(sample_data, method, experiment_cv_manager, X_train)
                prop_counterfactual = None

                if len(covered_samples_class) > 0:
                    counts = covered_samples_class.value_counts().to_dict()

                    if method == 'lore':
                        foil_class = sample_data['counterfactual_sample']['y']
                    else:
                        foil_class = sample_data['counterfactual_class']

                    if foil_class in counts:
                        prop_counterfactual = counts[foil_class] / len(covered_samples_class)

                elif method == 'rf_ocse':
                    prop_counterfactual = 1

                sample_res = {
                    'cv': experiment_cv_manager.cv,
                    'sample_id': sample_id,
                    'dataset': experiment_cv_manager.dataset_name,
                    ('prop_populated', method): min(1, len(covered_samples_class)),
                    ('prop_covered_samples', method): len(covered_samples_class) / len(X_train),
                    ('prop_counterfactual', method): prop_counterfactual
                }

                samples.append(sample_res)

    return samples


def process_dataset_cv(args):
    dataset, cv, methods = args
    methods_samples = defaultdict(dict)
    cv_manager = ExperimentCVManager.read_experiment(os.path.join(BASE_PATH, dataset, cv))

    for method in methods:
        for sample in process_dataset_method_cv(cv_manager, method):
            key = (sample['cv'], sample['sample_id'], sample['dataset'])
            methods_samples[key].update(sample)

    return list(methods_samples.values())


def generate_df(datasets, cvs, methods):
    args = list(product(datasets, cvs, [methods]))

    with Pool() as pool:
        samples = pool.map(process_dataset_cv, args)

    return list(chain(*samples))


if __name__ == '__main__':

    result_df = pd.DataFrame(generate_df(datasets, cvs, methods))
    result_df.set_index(['cv', 'sample_id', 'dataset'], inplace=True)
    result_df.columns = pd.MultiIndex.from_tuples([(col,) if isinstance(col, str) else col for col in result_df.columns])
    result_df.sort_index(axis=1, level=[0, 1], inplace=True)

    pd.set_option('display.max_columns', 100)

    df_dataset = result_df.groupby(level=2).mean()
    df_dataset = (df_dataset * 100).round(2)


    df_res = pd.DataFrame()

    for method in methods:
        df_res[method, 's-fidelity'] = df_dataset['prop_counterfactual', method]
        df_res[method, 'set coverage'] =  df_dataset['prop_covered_samples', method].astype(str) + \
                                          " (" + df_dataset['prop_populated', method].astype(str) + "%)"

    df_res.columns = pd.MultiIndex.from_tuples(df_res.columns)


    df_res = df_res[sorted(methods)]
    df_res.to_csv('research_paper/reports/cs_table.csv')


    def store_latex(df, file):
        with open('research_paper/reports/' + file, 'w') as fout:
            fout.write(df.to_latex())


    store_latex(df_res, 'cs_table.latex')