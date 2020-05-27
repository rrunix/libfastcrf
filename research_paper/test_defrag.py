from research_paper.methods.defragtrees.defragtrees import DefragModel
import pandas as pd
import pprint
from sklearn.ensemble import RandomForestClassifier
import math
import numpy as np
np.set_printoptions(suppress=True)
from research_paper import dataset_reader

# dataset_info, X_prep, y = dataset_reader.load_dataset('wine', use_one_hot=False, cache=False)
#
# from sklearn.model_selection import train_test_split

import resource
resource.setrlimit(resource.RLIMIT_AS, (4 * 1073741824, 6 * 1073741824))

base_path = 'research_paper/experiments/adult/cv_0/'

X_train = pd.read_csv(base_path + 'x_train.csv', index_col=None).values
y_train = pd.read_csv(base_path + 'y_train.csv', index_col=None, names=['y'])
y_train = y_train[y_train.columns[0]].values

# X_prep = X_prep.values
# X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.1, random_state=0)

rf = RandomForestClassifier(max_depth=None, n_estimators=10, random_state=0)
rf.fit(X_train, y_train)

Kmax = 10
splitter = DefragModel.parseSLtrees(rf) # parse sklearn tree ensembles into the array of (feature index, threshold)
mdl = DefragModel(modeltype='classification', maxitr=100, qitr=0, tol=1e-6, restart=20, verbose=0)
# print(splitter.nbytes)
mdl.fit(X_train, y_train, splitter, Kmax, fittype='FAB')


def parse_rule(rule_id, mdl):
    bounds =  {feat: [None, None] for feat in range(len(mdl.featurename_))}
    for feat, cmp, value in mdl.rule_[rule_id]:
        feat_bounds = bounds[feat-1]
        if cmp == 1:
            feat_bounds[0] = value if feat_bounds[0] is None else max(value, feat_bounds[0])
        else:
            feat_bounds[1] = value if feat_bounds[1] is None else min(value, feat_bounds[1])

    return {'class': mdl.pred_[rule_id], 'bounds': bounds}


def generate_otherwise_rules(rule, base_class):
    rules = []
    if rule['class'] != base_class:
        feats = list(rule['bounds'].keys())

        def generate(previous_conditions, feat_idx):
            res = []

            if feat_idx < len(feats):
                lq, gt = rule['bounds'][feat_idx]

                if gt:
                    res = generate(previous_conditions + [{'feat': feat_idx, 'value': gt, 'is_leq': True}], feat_idx + 1)

                if lq:
                    res = generate(previous_conditions + [{'feat': feat_idx, 'value': lq, 'is_leq': False}], feat_idx + 1)

            return res

        rules = generate([], 0)

    return rules

rules = [parse_rule(rule_id, mdl) for rule_id in range(len(mdl.rule_))]



