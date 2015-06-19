#!/usr/bin/env python

"""


"""

__depends__ = ["human", "model_belief_by_trial_fit.csv"]
__random__ = True

import os
import util
import pandas as pd
import numpy as np


def run(dest, data_path, results_path, seed):
    np.random.seed(seed)

    cols = ['likelihood', 'counterfactual', 'model', 'fitted', 'version']

    # load human data
    human = util.load_human(data_path)['C']\
        .dropna(axis=0, subset=['mass? response'])
    human.loc[:, 'mass? response'] = (human['mass? response'] + 1) / 2.0
    human = human[['version', 'kappa0', 'stimulus', 'mass? response', 'pid']]\
        .rename(columns={'mass? response': 'h'})

    # load model data
    model = pd.read_csv(os.path.join(results_path, 'model_belief_by_trial_fit.csv'))
    model = model[cols + ['kappa0', 'stimulus', 'pid', 'p']]\
        .rename(columns={'p': 'm'})
    model.loc[:, 'm'] = (model['m'] > 0.5).astype(int)

    data = pd.merge(human, model).set_index(cols + ['kappa0', 'stimulus', 'pid'])
    TP = ((data['h'] == 1) & (data['m'] == 1)).groupby(level=cols).sum()
    FP = ((data['h'] == 0) & (data['m'] == 1)).groupby(level=cols).sum()
    TN = ((data['h'] == 0) & (data['m'] == 0)).groupby(level=cols).sum()
    FN = ((data['h'] == 1) & (data['m'] == 0)).groupby(level=cols).sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (TP + TN) / (TP + FP + FN + TN)

    results = pd.DataFrame({
        'precision': precision,
        'recall': recall,
        'F1': F1,
        'accuracy': accuracy
    })

    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.data_path, args.results_path, args.seed)
