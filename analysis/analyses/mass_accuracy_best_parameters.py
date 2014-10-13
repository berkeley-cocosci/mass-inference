#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
import re
from path import path


def run(results_path, seed):
    np.random.seed(seed)

    human = pd.read_csv(path(results_path).dirname().joinpath(
        "mass_responses_by_stimulus.csv"))

    human = human\
        .set_index(['species', 'class', 'version', 'kappa0', 'stimulus'])\
        .ix[('human', 'static', 'H')]['median']\
        .sortlevel()

    store = pd.HDFStore(path(results_path).dirname().joinpath(
        'model_belief_fit.h5'))

    query = util.get_query()
    params = store["/{}/ipe/param_ref".format(query)]
    store_pth = r"/{}/ipe/(params_[0-9]+)/static/belief".format(query)
    model_belief = []
    for key in store.keys():
        matches = re.match(store_pth, key)
        if not matches:
            continue
        df = store[key]\
            .reset_index()\
            .groupby('version')\
            .get_group('H')\
            .drop('version', axis=1)
        sigma, phi = params.ix[matches.group(1)]
        df['sigma'] = sigma
        df['phi'] = phi
        model_belief.append(df)

    store.close()

    model = pd.concat(model_belief)\
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
