import pandas as pd

import pandas as pd
import glob
import os
from collections import defaultdict
import ast


series = {}
for dataset in ['credit', 'adult', 'compass']:
    dataset_model_distance = '_experiments/ok/*__{}__forest__one_norm__rf_ocre*'.format(dataset)

    experiments = glob.glob(dataset_model_distance)

    sample_info_counterfactuals = defaultdict(dict)
    method_names = []
    for experiment in experiments:
        if 'nolazy' in experiment:
            continue

        experiment_info = experiment.split('__')
        date = experiment_info[0].replace('_experiments/', '')
        dataset = experiment_info[1]
        model = experiment_info[2]
        distance = experiment_info[3]
        method = experiment_info[4]
        method_names.append(method)

        with open(os.path.join(experiment, 'minimum_distances.txt')) as fin:
            file_content = fin.read()

        file_content = file_content.replace('inf', '-1')
        counterfactuals_info = ast.literal_eval(file_content)


        for sample, info in counterfactuals_info.items():
            p0, p1 = info['counterfactual_proba']
            sample_info_counterfactuals[sample][(method, 'p0')] = p0
            sample_info_counterfactuals[sample][(method, 'p1')] = p1

    report = pd.DataFrame(sample_info_counterfactuals).T
    series[dataset] = report.mean()

print(pd.DataFrame(series))