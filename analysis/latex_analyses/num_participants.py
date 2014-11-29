#!/usr/bin/env python

"""
Produce a LaTeX file that includes details about how participants were excluded,
e.g. how many failed the posttest, how many had duplicate trials, etc. This
expects the file "num_participants.csv" to be present in the results directory.
"""

import sys
import util
import pandas as pd


def run(latex_path, results_path):
    results = pd.read_csv(
        results_path.joinpath("num_participants.csv"))

    results = results\
        .set_index('version')\
        .stack()

    replace = {
        'H': 'One',
        'G': 'Two',
        'I': 'Three'
    }

    fh = open(latex_path, "w")

    for (version, note), num in results.iteritems():
        note = ''.join(map(lambda x: x.capitalize(), note.split('_')))
        cmdname = "Exp{}{}".format(replace[version], note)
        fh.write(util.newcommand(cmdname, int(num)))

    fh.close()
    return latex_path


if __name__ == "__main__":
    parser = util.default_argparser(__doc__)
    args = parser.parse_args()
    run(args.latex_path, args.results_path)
