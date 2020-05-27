import pandas as pd
import json
import os
from itertools import product
from collections import defaultdict
import decimal

experiments_path = 'research_paper/experiments/'
methods = ['rf_ocse', 'lore', 'mace', 'mo', 'rf_ocse', 'fbt']
datasets = ['abalone',
            'adult',
            'banknote',
            'compas',
            'credit',
            'mammographic_mases', 'occupancy', 'pima', 'postoperative', 'seismic']

cvs = ['cv_' + str(i) for i in range(10)]
precision = decimal.Decimal('.0000001')


def truncate_distance(distance):
    if distance is not None:
        decimal_val = decimal.Decimal(str(distance)).quantize(
            precision,
            rounding=decimal.ROUND_DOWN
        )
        distance = float(decimal_val)
    return distance


def load_cv_method(dataset, cv, method):
    base_path = os.path.join(experiments_path, dataset, cv, method)
    samples = []

    if os.path.exists(base_path):
        for sample_file in os.listdir(base_path):
            if sample_file.startswith('sample_') and sample_file.endswith('.json'):
                sample_id = sample_file[:-5]

                with open(os.path.join(base_path, sample_file)) as fin:
                    sample_data = json.load(fin)

                is_valid = sample_data.get('valid_', None) or sample_data.get('counterfactual_found', True)

                if not is_valid:
                    distance = None
                elif 'counterfactual_distance' in sample_data:
                    distance = truncate_distance(sample_data['counterfactual_distance'])
                elif 'distance':
                    distance = truncate_distance(sample_data['distance'])

                if 'counterfactual_time' in sample_data:
                    time = sample_data['counterfactual_time']
                else:
                    time = sample_data['extraction_time']

                samples.append({
                    'cv': cv,
                    'sample_id': sample_id,
                    'dataset': dataset,
                    ('time', method): time,
                    ('valid', method): is_valid,
                    ('distance', method): distance
                })

    return samples


def process_dataset_cv(dataset):
    methods_samples = []

    for method, cv in product(methods, cvs):
        methods_samples.extend(load_cv_method(dataset, cv, method))

    return methods_samples


def generate_df():
    all_samples = defaultdict(dict)
    for dataset in datasets:
        for sample in process_dataset_cv(dataset):
            key = (sample['cv'], sample['dataset'], sample['sample_id'])
            all_samples[key] = {**all_samples[key], **{k: sample[k] for k in sample.keys()
                                                       if 'valid' in k or 'distance' in k or 'time' in k}}

    return all_samples


result_df = pd.DataFrame(generate_df()).T
result_df.columns = pd.MultiIndex.from_tuples([(col, ) if isinstance(col, str) else col for col in result_df.columns])
distance_cols = [col for col in result_df.columns if 'distance' in col]
valid_cols = [col for col in result_df.columns if 'valid' in col]
time_cols = [col for col in result_df.columns if 'time' in col]

result_df[distance_cols] = result_df[distance_cols].astype(float)
result_df[time_cols] = result_df[time_cols].astype(float)


for valid_col in valid_cols:
    result_df[valid_col] = pd.to_numeric(result_df[valid_col], errors='coerce')
    result_df[valid_col].fillna(0, inplace=True)


optimum_distance = result_df[('distance', 'rf_ocse')]

for col in result_df['distance'].columns:
    result_df['rci', col] = (100 * (1 - (optimum_distance / result_df['distance', col]).clip(0, 1))).round(3)



pd.set_option('display.max_columns', 100)
df_mean = result_df.groupby(level=1).mean()
df_mean.sort_index(axis=1, level=0, inplace=True)
df_mean.fillna(0, inplace=True)
df_mean = df_mean.round(2)


def store_latex(df, file):
    with open('research_paper/reports/' + file, 'w') as fout:
        fout.write(df.to_latex())

# df_times = df_mean[['time']]
# df_times.to_csv('research_paper/reports/c_times.csv')
# store_latex(df_times, 'c_times.latex')
#
#
# df_mean_distance_valid = df_mean[['valid', 'rci']].copy()
# df_mean_distance_valid['valid'] = (df_mean_distance_valid['valid'] * 100).astype(int)
# df_mean_distance_valid.columns = df_mean_distance_valid.columns.swaplevel(0, 1)
# df_mean_distance_valid_join = df_mean_distance_valid.groupby(axis=1, level=0).apply(lambda x: x[x.columns[0][0], 'rci'].astype(str)
#                                                                                   + " (" +
#                                                                                   x[x.columns[0][0], 'valid'].astype(str)
#                                                                                   + "%)")

# df_mean_distance_valid_join.to_csv('research_paper/reports/c_distances_valid.csv')
# store_latex(df_mean_distance_valid_join, 'c_distances_valid.latex')
#
#
# df_all = result_df[['valid', 'rci']].copy()
#
# for method in methods:
#     df_all.loc[df_all['valid', method] == 0, ('rci', method)] = None
#
#
# df_mean_all = df_all.mean()
# df_std_all = df_all.std()
#
#
# print(df_mean_all)
# print(df_std_all)