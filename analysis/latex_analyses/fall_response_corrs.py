#!/usr/bin/env python

import os
import util
import pandas as pd


def run(dest, results_path):
    results = pd.read_csv(os.path.join(results_path, "fall_response_corrs.csv"))

    results = results.set_index(['block', 'X', 'Y'])

    fh = open(dest, "w")

    for (block, x, y), corrs in results.iterrows():
        cmdname = "FallCorr{}{}{}".format(x, y, block)
        cmd = util.latex_pearson.format(**corrs)
        fh.write(util.newcommand(cmdname, cmd))

    fh.close()


if __name__ == "__main__":
    parser = util.default_argparser(__doc__)
    args = parser.parse_args()
    run(args.dest, args.results_path)
