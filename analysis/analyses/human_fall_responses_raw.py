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
    pid (string)
        participant id
    fall? response
        normalized response to the fall? query

"""

__depends__ = ["human"]

import util
import pandas as pd
import numpy as np


def run(dest, data_path):
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

            human = human_all.set_index(['kappa0', 'stimulus', 'pid'])[['fall? response']]
            human = ((human - 1) / 6.0).reset_index()
            human['block'] = block
            human['version'] = version
            results.append(human)

    results = pd.concat(results)\
                .set_index(['version', 'block', 'kappa0', 'stimulus', 'pid'])\
                .sortlevel()

    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.data_path)

