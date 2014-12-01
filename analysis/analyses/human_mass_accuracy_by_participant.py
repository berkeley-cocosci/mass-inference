#!/usr/bin/env python

"""
Computes accuracy for each participant on "which is heavier?". Produces a csv
file with the following columns:

    version (string)
        experiment version
    kappa0 (float)
        true log mass ratio
    num_mass_trials (int)
        the number of trials on which participants were asked "which is heavier?"
    pid (string)
        unique participant id
    mass? correct (float)
        participant accuracy across stimuli

"""

__depends__ = ["human"]
__random__ = True

import util
import numpy as np

def run(dest, data_path, seed):
    np.random.seed(seed)
    human = util.load_human(data_path)['C']\
        .dropna(axis=0, subset=['mass? response'])

    results = human\
        .groupby(['version', 'kappa0', 'num_mass_trials', 'pid'])['mass? correct']\
        .mean()\
        .to_frame()

    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.data_path, args.seed)
