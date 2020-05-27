import os
from research_paper import dataset_reader
import tqdm
import json
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

from research_paper.experiment_manager import ExperimentCVManager


def create_experiment(dataset_name, num_folds, rf_params=None, dataset_params=None, base_path='research_paper/experiments/'):
    rf_params = rf_params or {}
    dataset_params = dataset_params or {}

    experiment_path = os.path.join(base_path, dataset_name)

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

        dataset_info, X, y = dataset_reader.load_dataset(dataset_name, **dataset_params)

        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
        folds = kfold.split(X.index)

        with open(os.path.join(experiment_path, 'params.json'), 'w') as fout:
            json.dump({
                'rf_params': rf_params,
                'dataset_params': dataset_params,
                'num_folds': num_folds,
                'dataset_name': dataset_name
            }, fout, indent=3)

        for fold_id, (train_idx, test_idx) in tqdm.tqdm(enumerate(folds)):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]

            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            rf = RandomForestClassifier(**rf_params)
            rf.fit(X_train, y_train)

            fold_path = os.path.join(experiment_path, 'cv_{cv}'.format(cv=fold_id))

            ExperimentCVManager.bootstrap_cv_experiment(fold_path, X_train, y_train, X_test, y_test, rf, dataset_info)


if __name__ == '__main__':
    num_folds = 10
    rf_params = {'n_estimators': 10}
    dataset_params = {'max_size': 10_000}

    for dataset_name in dataset_reader.dataset_list:
        create_experiment(dataset_name, num_folds, rf_params=rf_params, dataset_params=dataset_params)
