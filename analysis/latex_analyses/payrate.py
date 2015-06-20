#!/usr/bin/env python

"""
Produce a LaTeX file that includes details about how long it took participants
to complete the experiments.
"""

__depends__ = ["payrate.csv"]

import os
import util
import pandas as pd


def run(dest, results_path):
    results = pd.read_csv(os.path.join(results_path, "payrate.csv")).set_index('version').T

    replace = {
        'H': 'One',
        'G': 'Two',
        'I': 'Three'
    }

    fh = open(dest, "w")

    for version, data in results.iteritems():
        cmdname = "Exp{}MedianTime".format(replace[version])
        time = "{:.1f}".format(pd.to_timedelta(data['median_time']).total_seconds() / 60.0)
        fh.write(util.newcommand(cmdname, time))

    fh.close()


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path)
