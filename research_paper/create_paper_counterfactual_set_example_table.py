import json
import pandas as pd
import os
import pprint
import math

from research_paper.create_paper_counterfactual_sets_table import get_bounds
from research_paper.experiment_manager import ExperimentCVManager

methods = ['lore', 'rf_ocse', 'fbt']


experiment = 'research_paper/experiments/adult/cv_0'
observation = 'sample_0.json'

experiment_manager = ExperimentCVManager.read_experiment(experiment)
dataset_info = experiment_manager.datataset_info
factual_sample = None

methods_data = {}

adult_mappings = {
    'NativeCountry': {0: 'United-States',
                      1: 'Non-United-Stated'},
    'WorkClass': {0: 'Federal-gov',
                  1: 'Local-gov',
                  2: 'Private',
                  3: 'Self-emp-inc',
                  4: 'Self-emp-not-inc',
                  5: 'State-gov',
                  6: 'Without-pay'},
    'EducationLevel': {0: 'prim-middle-school',
                       1: 'high-school',
                       2: 'HS-grad',
                       3: 'Some-college',
                       4: 'Bachelors',
                       5: 'Masters',
                       6: 'Doctorate',
                       7: 'Assoc-voc',
                       8: 'Assoc-acdm',
                       9: 'Prof-school'},
    'MaritalStatus': {
        0: 'Divorced',
        1: 'Married-AF-spouse',
        2: 'Married-civ-spouse',
        3: 'Married-spouse-absent',
        4: 'Never-married',
        5: 'Separated',
        6: 'Widowed'},

    'Occupation': {
        0: 'Adm-clerical',
        1: 'Armed-Forces',
        2: 'Craft-repair',
        3: 'Exec-managerial',
        4: 'Farming-fishing',
        5: 'Handlers-cleaners',
        6: 'Machine-op-inspct',
        7: 'Other-service',
        8: 'Priv-house-serv',
        9: 'Prof-specialty',
        10: 'Protective-serv',
        11: 'Sales',
        12: 'Tech-support',
        13: 'Transport-moving',
    },
    'Relationship': {
        0: 'Husband',
        1: 'Not-in-family',
        2: 'Other-relative',
        3: 'Own-child',
        4: 'Unmarried',
        5: 'Wife'}}

for method in methods:
    sample_file_name = os.path.join(experiment, method, observation)
    with open(sample_file_name, 'r') as fin:
        sample_data = json.load(fin)

    bounds = get_bounds(sample_data, method, experiment_manager)

    print(bounds, method)

    for feat, (gt, lt) in bounds.items():
        if feat in adult_mappings:
            values = [adult_mappings[feat][v] for v in sorted(adult_mappings[feat].keys())]

            if gt is not None:
                gt = int(math.floor(gt + 1))
                values = values[gt:]

            if lt is not None:
                lt = int(math.floor(lt))
                values = values[:lt]

            if gt is None and lt is None:
                values = ['-']

            bounds[feat] = values

    methods_data[method] = bounds


print(pd.DataFrame(methods_data).to_latex())

