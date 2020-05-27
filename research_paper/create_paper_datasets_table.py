from research_paper import dataset_reader

import pprint

for dataset_name in sorted(dataset_reader.dataset_list.keys()):
    dataset_info, X, _ = dataset_reader.load_dataset(dataset_name, cache=False, use_one_hot=True)
    feature_types = {attr_info['type'] for attr_info in dataset_info.dataset_description.values()}
    pprint.pprint({
        'name': dataset_name,
        'n_instances': len(X),
        'feature_types': feature_types,
        'n_features': len(dataset_info.dataset_description.values())
    })