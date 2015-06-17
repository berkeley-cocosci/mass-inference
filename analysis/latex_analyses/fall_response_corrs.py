#!/usr/bin/env python

"""
Produces a LaTeX file with the correlations between model and human fall
responses.
"""

__depends__ = ["fall_response_corrs.csv"]

import os
import util
import pandas as pd


def run(dest, results_path):
    main_query = util.load_query()
    results = pd\
        .read_csv(os.path.join(results_path, "fall_response_corrs.csv"))\
        .set_index(['query', 'block', 'X', 'Y'])

    format_pearson = util.load_config()["latex"]["pearson"]

    fh = open(dest, "w")

    for (query, block, x, y), corrs in results.iterrows():
        cmdname = "FallCorr{}{}{}".format(x, y, block)
        if query != main_query:
            cmdname += "".join([x.capitalize() for x in query.split("_")])
        cmd = format_pearson.format(**corrs)
        fh.write(util.newcommand(cmdname, cmd))

    fh.close()


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path)
