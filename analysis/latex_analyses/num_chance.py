#!/usr/bin/env python

"""
Produces a LaTeX file with the number of stimuli for which people were not
significantly above chance in judging which color is heavier, for each of the
different versions of the experiment. Expects the file "num_chance.csv" to be
present in the results directory.
"""

import os
import util
import pandas as pd


def run(dest, results_path):
    results = pd.read_csv(os.path.join(results_path, ("num_chance.csv")))

    alpha = results.columns[-1]
    results = results\
        .groupby('version')\
        .sum()[alpha]

    replace = {
        'H': 'One',
        'G': 'Two',
        'I': 'Three'
    }

    fh = open(dest, "w")

    for version, num in results.iteritems():
        name = "MassAccNumChanceExp{}".format(replace[version])
        fh.write(util.newcommand(name, int(num)))
        fh.write(util.newcommand("{}Correction".format(name), alpha))

    fh.close()


if __name__ == "__main__":
    parser = util.default_argparser(__doc__)
    args = parser.parse_args()
    run(args.dest, args.results_path)
