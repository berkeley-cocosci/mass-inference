#!/usr/bin/env python

"""
Computes the average judgments to "will it fall?" for both participants and the
model. Produces a csv file with the following columns:

    version (string)
        the experiment version
    block (string)
        the experiment phase
    species (string)
        either 'human' or 'model'
    kappa0 (float)
        the true log mass ratio
    stimulus (string)
        the name of the stimulus
    lower (float in [0, 1])
        lower bound of the 95% confidence interval
    median (float in [0, 1])
        median value across the bootstrapped means
    upper (float in [0, 1])
        upper bound of the 95% confidence interval

"""

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
                .apply(lambda x: util.bootstrap_mean((x - 1) / 6.0))\
                .unstack(-1)\
                .reset_index()
            human['species'] = 'human'
            human['block'] = block
            human['version'] = version
            results.append(human)

            kappas = list(human['kappa0'].unique()) + [0.0]
            model = data['ipe'][block]\
                .P_fall_stats[kappas]\
                .stack()\
                .unstack('stat')\
                .reset_index()\
                .rename(columns={
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
    parser = util.default_argparser(__doc__)
    args = parser.parse_args()
    run(args.results_path, args.seed)

