#!/usr/bin/env python

"""
Produces a LaTeX file with the log likelihood ratio of the learning model to the
static model for each likelihood type and experiment version.
"""

__depends__ = ["model_log_lh_ratios.csv"]

import os
import util
import pandas as pd


def run(dest, results_path, counterfactual):
    results = pd\
        .read_csv(os.path.join(results_path, 'model_log_lh_ratios.csv'))\
        .groupby(['counterfactual', 'fitted'])\
        .get_group((counterfactual, True))\
        .set_index(['likelihood', 'version', 'num_mass_trials'])

    replace = {
        'G': 'ExpTwo',
        'H': 'ExpOne',
        'I': 'ExpThree',
        1: 'OneTrial',
        2: 'TwoTrials',
        3: 'ThreeTrials',
        4: 'FourTrials',
        5: 'FiveTrials',
        -1: 'AcrossSubjs',
        8: '',
        20: ''
    }

    query = util.load_query()

    fh = open(dest, "w")

    for (lh, version, num_trials), llhr in results.iterrows():
        if lh == "ipe_" + query:
            lh = 'Ipe'
        else:
            lh = "".join([x.capitalize() for x in lh.split("_")])

        cmdname = "llhr{}{}{}".format(lh.capitalize(), replace[version], replace[num_trials])
        cmd = r"$\textrm{{LLR}}={llhr:.2f}$".format(**llhr)
        fh.write(util.newcommand(cmdname, cmd))

    fh.close()


if __name__ == "__main__":
    config = util.load_config()
    parser = util.default_argparser(locals())
    if config['analysis']['counterfactual']:
        parser.add_argument(
            '--no-counterfactual',
            action='store_false',
            dest='counterfactual',
            default=True,
            help="don't plot the counterfactual likelihoods")
    else:
        parser.add_argument(
            '--counterfactual',
            action='store_true',
            dest='counterfactual',
            default=False,
            help='plot the counterfactual likelihoods')
    args = parser.parse_args()
    run(args.to, args.results_path, args.counterfactual)
