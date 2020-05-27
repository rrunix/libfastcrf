import glob
import json
import pandas as pd

datasets_depth = []

for explanation_file in glob.glob('research_paper/experiments/*/*/rf_ocse/*.json'):
    with open(explanation_file, 'r') as fin:
        sample_data = json.load(fin)

    if 'splits' not in sample_data:
        # Counterfactual was estimated using mo approach
        continue

    dataset = explanation_file.split("/")[2]
    datasets_depth.append({'dataset': dataset, 'depth': len(sample_data['splits'])})


df_depth = pd.DataFrame(datasets_depth)
print(df_depth.groupby('dataset').mean())
print(df_depth.mean())