#!/usr/bin/env python

"""
Computes the average participant judgments to "will it fall?". Produces a csv
file with the following columns:

    version (string)
        the experiment version
    block (string)
        the experiment phase
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
    N (int)
        how many samples the mean was computed over

"""

__depends__ = ["human"]

import util
import pandas as pd
import numpy as np


def run(dest, data_path, seed):
    np.random.seed(seed)
    data = util.load_human(data_path)

    results = []
    for block in ['A', 'B']:
        versions = list(data[block]['version'].unique())
        versions.extend(['GH', 'all'])
        for version in versions:
            if version == 'all':
                human_all = data[block]
            elif version == 'GH':
                human_all = data[block]\
                    .set_index('version')\
                    .groupby(lambda x: x in ('G', 'H'))\
                    .get_group(True)\
                    .reset_index()
            else:
                human_all = data[block]\
                    .groupby('version')\
                    .get_group(version)

            human = human_all\
                .groupby(['kappa0', 'stimulus'])['fall? response']\
                .apply(lambda x: util.bootstrap_mean((x - 1) / 6.0))\
                .unstack(-1)\
                .reset_index()
            human['block'] = block
            human['version'] = version
            results.append(human)

    results = pd.concat(results)\
                .set_index(['version', 'block', 'kappa0', 'stimulus'])\
                .sortlevel()

    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals(), seed=True)
    args = parser.parse_args()
    run(args.dest, args.data_path, args.seed)

