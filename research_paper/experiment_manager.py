import json
import os
import numpy as np
import pickle
import pandas as pd
import bz2
from research_paper.logger import get_logger

try:
  from pathlib import Path
except ImportError:
  from pathlib2 import Path


class NpEncoder(json.JSONEncoder):
    # https://stackoverflow.com/a/57915246
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(NpEncoder, self).default(obj)


class ExperimentCVManager:

    def __init__(self, experiment_path, X_train, y_train, X_test, y_test, rf, dataset_info):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.experiment_path = experiment_path
        self.rf = rf
        self.datataset_info = dataset_info
        self.dataset_name = experiment_path.split("/")[2]
        self.cv = experiment_path.split("/")[3]

    @staticmethod
    def read_experiment(experiment_path):

        with bz2.BZ2File(os.path.join(experiment_path, 'model.pickle'), 'rb') as fin:
            rf = pickle.load(fin)

        X_train = pd.read_csv(os.path.join(experiment_path, 'x_train.csv'), index_col=None)
        X_test = pd.read_csv(os.path.join(experiment_path, 'x_test.csv'), index_col=None)
        y_train = pd.read_csv(os.path.join(experiment_path, 'y_train.csv'), index_col=None, names=['y'])
        y_test = pd.read_csv(os.path.join(experiment_path, 'y_test.csv'), index_col=None, names=['y'])

        y_train = y_train[y_train.columns[0]]
        y_test = y_test[y_test.columns[0]]

        with open(os.path.join(experiment_path, 'dataset_info.pickle'), 'rb') as fin:
            dataset_info = pickle.load(fin)

        return ExperimentCVManager(experiment_path, X_train, y_train, X_test, y_test, rf, dataset_info)

    @staticmethod
    def bootstrap_cv_experiment(experiment_path, X_train, y_train, X_test, y_test, rf, dataset_info):
        os.makedirs(experiment_path, exist_ok=False)

        with bz2.BZ2File(os.path.join(experiment_path, 'model.pickle'), 'wb') as fout:
            pickle.dump(rf, fout, protocol=2)

        X_train.to_csv(os.path.join(experiment_path, 'x_train.csv'), index=False)
        X_test.to_csv(os.path.join(experiment_path, 'x_test.csv'), index=False)
        y_train.to_csv(os.path.join(experiment_path, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(experiment_path, 'y_test.csv'), index=False)

        with open(os.path.join(experiment_path, 'dataset_info.pickle'), 'wb') as fout:
            pickle.dump(dataset_info, fout, protocol=2)

    def create_method_logger(self, method_name, params):
        return MethodExperimentManager(self.experiment_path, method_name, self, **params)


class MethodExperimentManager:

    def __init__(self, base_path, method_name, experiment_manager, overwrite=False):
        self.method_name = method_name
        self.experiment_method_path = os.path.join(base_path, method_name)
        Path(self.experiment_method_path).mkdir(exist_ok=True)
        self.experiment_manager = experiment_manager
        self.logger = get_logger(self)
        self.log_stdout = open(os.path.join(self.experiment_method_path, 'stout.txt'), 'w')
        self.overwrite = overwrite

    def log_configuration(self, configuration):
        pass

    def get_sample_filename(self, idx):
        return os.path.join(self.experiment_method_path, 'sample_{idx}.json'.format(idx=idx))

    def has_been_processed(self, idx):
        return os.path.exists(self.get_sample_filename(idx))

    def log_observation_result(self, idx, result):
        with open(self.get_sample_filename(idx), 'w') as fout:
            json.dump(result, fout, indent=3, cls=NpEncoder)

    def log_observation_error(self, idx, error):
        self.logger.exception("On observation {idx}".format(idx=idx))

    def finish_experiment(self):
        self.log_stdout.close()

    def get_tmp_folder(self):
        return 'tmp/'
