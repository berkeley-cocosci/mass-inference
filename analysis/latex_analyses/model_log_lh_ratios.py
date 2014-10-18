#!/usr/bin/env python

import sys
import util
import pandas as pd


def run(latex_path, results_path):
    results = pd.read_csv(
        results_path.joinpath("model_log_lh_ratios.csv"))
    results = results.set_index(['version', 'num_trials'])

    replace = {
        'G': 'ExpTwoA',
        'H': 'ExpOne',
        'I': 'ExpTwoB',
        1: 'OneTrial',
        2: 'TwoTrials',
        3: 'ThreeTrials',
        4: 'FourTrials',
        5: 'FiveTrials',
        -1: 'AcrossSubjs',
        8: '',
        20: ''
    }

    fh = open(latex_path, "w")

    for (version, num_trials), llhr in results.iterrows():
        cmdname = "llhr{}{}".format(replace[version], replace[num_trials])
        cmd = "$D={llhr:.2f}$".format(**llhr)
        fh.write(util.newcommand(cmdname, cmd))

    fh.close()
    return latex_path


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
