#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "mass_responses_only_learned.csv"


def run(data, results_path, seed):
    np.random.seed(seed)
    results = []

    sp = pd\
        .read_csv(results_path.joinpath("switchpoint.csv"))
    bad_pids = sp['pid'][sp['20'] == 0]

    correct = data['human']['C']\
        .set_index('pid')\
        .drop(bad_pids)\
        .reset_index()
    correct = correct[
        ['kappa0', 'trial', 'mass? response', 'pid']].dropna()
    correct.loc[:, 'mass? response'] = [
        x * 2 - 1 for x in correct['mass? response'].copy()]
    correct['correct'] = correct['kappa0'] == correct['mass? response']
    acc = correct.groupby(['kappa0', 'trial'])['correct']\
                 .apply(util.beta)

    belief = pd.read_csv(results_path.joinpath('model_belief_agg.csv'))
    belief = belief.set_index('pid').drop(bad_pids).reset_index()
    avg_belief = belief\
        .groupby(['kappa0', 'trial'])['p']\
        .apply(util.bootstrap_mean)

    results = pd.DataFrame({
        'human': acc,
        'model': avg_belief,
    })
    results.columns.name = 'species'
    results = results\
        .unstack(-1)\
        .stack('species')\
        .reorder_levels(['kappa0', 'species', 'trial'])\
        .sortlevel()

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
