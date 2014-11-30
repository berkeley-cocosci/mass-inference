#!/usr/bin/env python

"""
Produce a LaTeX file that includes details about how participants were excluded,
e.g. how many failed the posttest, how many had duplicate trials, etc. This
expects the file "num_participants.csv" to be present in the results directory.
"""

__depends__ = ["num_participants.csv"]

import os
import util
import pandas as pd


def run(dest, results_path):
    results = pd.read_csv(os.path.join(results_path, "num_participants.csv"))

    results = results\
        .set_index('version')\
        .stack()

    replace = {
        'H': 'One',
        'G': 'Two',
        'I': 'Three'
    }

    fh = open(dest, "w")

    for (version, note), num in results.iteritems():
        note = ''.join([x.capitalize() for x in note.split('_')])
        cmdname = "Exp{}{}".format(replace[version], note)
        fh.write(util.newcommand(cmdname, int(num)))

    fh.close()


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.dest, args.results_path)
