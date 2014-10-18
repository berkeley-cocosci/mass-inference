#!/usr/bin/env python

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
        'G': 'TwoA',
        'I': 'TwoB'
    }

    fh = open(latex_path, "w")

    for (version, note), num in results.iteritems():
        note = ''.join(map(lambda x: x.capitalize(), note.split('_')))
        cmdname = "Exp{}{}".format(replace[version], note)
        fh.write(util.newcommand(cmdname, int(num)))

    fh.close()
    return latex_path


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
