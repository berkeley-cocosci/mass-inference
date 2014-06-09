#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "fall_responses.csv"


def run(data, results_path, version, seed):
    np.random.seed(seed)
    results = []

    for block in ['A', 'B']:
        human = data['human'][block]\
            .groupby(['kappa0', 'stimulus'])['fall? response']\
            .apply(util.bootstrap_mean)\
            .unstack(-1)\
            .reset_index()
        human['species'] = 'human'
        human['block'] = block
        results.append(human)

        kappas = list(human['kappa0'].unique()) + [0.0]
        model = data['ipe'][block]\
            .P_fall_smooth[kappas]\
            .stack()\
            .reset_index()\
            .rename(columns={
                0: 'median',
                'kappa': 'kappa0'
            })
        model['species'] = 'model'
        model['block'] = block
        results.append(model)

    results = pd.concat(results)\
                .set_index(['block', 'species', 'kappa0', 'stimulus'])\
                .sortlevel()

    pth = results_path.joinpath(version, filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
