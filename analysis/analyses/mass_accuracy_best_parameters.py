#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path


def run(results_path, seed):
    np.random.seed(seed)

    human = pd.read_csv(path(results_path).dirname().joinpath(
        "mass_responses_by_stimulus.csv"))

    human = human\
        .set_index(['species', 'class', 'version', 'kappa0', 'stimulus'])\
        .ix[('human', 'static', 'H')]['median']\
        .sortlevel()

    model_belief = pd.read_csv(path(results_path).dirname().joinpath(
        'model_belief_agg_all_params.csv'))

    query = util.get_query()
    model = model_belief\
        .groupby(['likelihood', 'model', 'query', 'version'])\
        .get_group(('ipe', 'static', query, 'H'))\
        .groupby(['sigma', 'phi', 'kappa0', 'stimulus'])['p correct']\
        .mean()

    results = []
    for (sigma, phi), model_df in model.groupby(level=['sigma', 'phi']):
        index = model_df.reset_index(['sigma', 'phi'], drop=True).index
        x = np.asarray(model_df)
        y = np.asarray(human.ix[index])
        err = pd.Series((x - y) ** 2, index=model_df.index)
        results.append(err)

    results = pd\
        .concat(results)\
        .reset_index(['sigma', 'phi'])\
        .rename(columns={0: 'sqerr'})

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
