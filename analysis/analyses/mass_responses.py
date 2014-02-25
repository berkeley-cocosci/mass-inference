#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "mass_responses.csv"


def run(data, results_path, seed):
    np.random.seed(seed)
    results = []

    models = pd.read_csv(results_path.joinpath("participant_fits.csv"))
    models = models\
        .set_index('pid')\
        .groupby('rank')['model']\
        .get_group(0)

    model_belief = pd.read_csv(
        results_path.joinpath('model_belief_agg.csv'))

    for model in list(models.unique()) + ['best']:
        if model == 'best':
            human = data['human']['C']\
                .dropna(axis=0, subset=['mass? response'])
            belief = model_belief\
                .set_index(['pid', 'model'])\
                .groupby(lambda x: models[x[0]] == x[1])\
                .get_group(True)\
                .reset_index()

        else:
            human = data['human']['C']\
                .dropna(axis=0, subset=['mass? response'])\
                .set_index('pid')\
                .ix[models.index[models == model]]\
                .reset_index()
            belief = model_belief\
                .groupby('model')\
                .get_group(model)

        human = human\
            .groupby(['kappa0', 'trial'])['mass? correct']\
            .apply(util.beta)\
            .unstack(-1)\
            .reset_index()
        human['class'] = model
        human['species'] = 'human'
        human = human\
            .set_index(['species', 'class', 'kappa0', 'trial'])\
            .stack()
        results.append(human)

        belief = belief\
            .groupby(['likelihood', 'kappa0', 'trial'])['p']\
            .apply(util.bootstrap_mean)\
            .unstack(-1)\
            .reset_index()\
            .rename(columns={'likelihood': 'species'})
        belief['class'] = model
        belief = belief\
            .set_index(['species', 'class', 'kappa0', 'trial'])\
            .stack()
        results.append(belief)

    results = pd.concat(results).unstack().sortlevel()

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
