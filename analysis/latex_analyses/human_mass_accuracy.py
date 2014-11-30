#!/usr/bin/env python

"""
Produces a latex file with the accuracy of human participants for each version
of the experiment.
"""

__depends__ = ["human_mass_accuracy.csv"]

import os
import util
import pandas as pd


def run(dest, results_path):
    results = pd\
        .read_csv(os.path.join(results_path, "human_mass_accuracy.csv"))\
        .set_index(['version', 'kappa0']) * 100

    latex_percent = util.load_config()["latex"]["percent"]

    replace = {
        'H': 'One',
        'G': 'Two',
        'I': 'Three',
        '-1.0': 'KappaLow',
        '1.0': 'KappaHigh',
        'all': ''
    }

    fh = open(dest, "w")

    for (version, kappa0), accuracy in results.iterrows():
        cmdname = "MassAccExp{}{}".format(
            replace[version], replace[kappa0])
        cmd = latex_percent.format(**accuracy)
        fh.write(util.newcommand(cmdname, cmd))

    fh.close()


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.dest, args.results_path)
