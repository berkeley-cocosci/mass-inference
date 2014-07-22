#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np


def run(results_path, seed):
    np.random.seed(seed)
    data = util.load_all()
    results = []

    for block in ['A', 'B']:
        versions = list(data['human'][block]['version'].unique())
        versions.extend(['GH', 'all'])
        for version in versions:
            if version == 'all':
                human_all = data['human'][block]
            elif version == 'GH':
                human_all = data['human'][block]\
                    .set_index('version')\
                    .groupby(lambda x: x in ('G', 'H'))\
                    .get_group(True)\
                    .reset_index()
            else:
                human_all = data['human'][block]\
                    .groupby('version')\
                    .get_group(version)

            human = human_all\
                .groupby(['kappa0', 'stimulus'])['fall? response']\
                .apply(lambda x: util.beta(x - 1, n=6))\
                .unstack(-1)\
                .reset_index()
            human['species'] = 'human'
            human['block'] = block
            human['version'] = version
            results.append(human)

            kappas = list(human['kappa0'].unique()) + [0.0]
            model = data['ipe'][block]\
                .P_fall_mean[kappas]\
                .stack()\
                .reset_index()\
                .rename(columns={
                    0: 'median',
                    'kappa': 'kappa0'
                })
            model['species'] = 'model'
            model['block'] = block
            model['version'] = version
            results.append(model)

    results = pd.concat(results)\
                .set_index(['version', 'block', 'species',
                            'kappa0', 'stimulus'])\
                .sortlevel()

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
