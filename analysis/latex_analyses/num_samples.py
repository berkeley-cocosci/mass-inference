#!/usr/bin/env python

"""
"""

__depends__ = ["num_samples.csv"]

import os
import util
import pandas as pd
import numpy as np


def run(dest, results_path, version, block, query):
    fits = pd.read_csv(os.path.join(results_path, "num_samples.csv"))\
        .groupby(['version', 'query', 'block'])\
        .get_group((version, query, block))\
        .set_index('k')
    best_k = fits['mse'].argmin()
    omega = np.sqrt(fits['intercept'][best_k])

    fh = open(dest, "w")
    fh.write(util.newcommand("BestFitK", r"$n={:d}$".format(int(best_k))))
    fh.write(util.newcommand("BestFitOmega", r"$\omega={:.2f}$".format(omega)))
    fh.close()


if __name__ == "__main__":
    config = util.load_config()
    parser = util.default_argparser(locals())
    parser.add_argument(
        '--version',
        default=config['analysis']['human_fall_version'],
        help='which version of the experiment to plot responses from')
    parser.add_argument(
        '--block',
        default='B',
        help='which block of the experiment to plot responses from')
    parser.add_argument(
        '--query',
        default=config['analysis']['query'],
        help='which ipe query to use')
    args = parser.parse_args()
    run(args.to, args.results_path, args.version, args.block, args.query)
