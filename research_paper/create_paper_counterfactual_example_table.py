import json
import pandas as pd
import os

from research_paper.experiment_manager import ExperimentCVManager

methods = ['ft', 'lore', 'mace', 'mo', 'rf_ocse', 'fbt']

experiment = 'research_paper/experiments/adult/cv_0'
observation = 'sample_0.json'

experiment_manager = ExperimentCVManager.read_experiment(experiment)
dataset_info = experiment_manager.datataset_info
factual_sample = None

methods_data = {}

for method in methods:
    sample_file_name = os.path.join(experiment, method, observation)
    with open(sample_file_name, 'r') as fin:
        sample_data = json.load(fin)

    counterfactual = sample_data['counterfactual_sample']

    is_valid = sample_data.get('valid_', None) or sample_data.get('counterfactual_found', True)

    if method == 'rf_ocse':
        counterfactual = {dataset_info.dataset_description[i]['name']: v for i, v in enumerate(counterfactual)}
        methods_data['factual_sample'] = {dataset_info.dataset_description[i]['name']: v for i, v in
                                          enumerate(sample_data['_observation'])}

    if 'y' in counterfactual:
        del counterfactual['y']

    print(method, is_valid)
    methods_data[method] = counterfactual

pd.set_option('display.max_columns', 100)
df = pd.DataFrame(methods_data)
df.sort_index(axis=1, inplace=True)
df_round = df.round(2)

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


df_with_names = df_round.copy().T

for col, mapping in adult_mappings.items():
    df_with_names[col] = df_with_names[col].astype(int)
    df_with_names = df_with_names.replace({col: mapping})

df_with_names = df_with_names.astype(str).T
df_with_names.sort_index(inplace=True)
print(df_with_names.to_latex())
