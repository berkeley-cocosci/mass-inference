#!/usr/bin/env python

"""
Produces a LaTeX file with the bayes factors of the learning model to the
static model for each likelihood type and experiment version.
"""

__depends__ = ["bayes_factors.csv"]

import os
import util
import pandas as pd


def run(dest, results_path, counterfactual):
    results = pd\
        .read_csv(os.path.join(results_path, 'bayes_factors.csv'))\
        .groupby('counterfactual')\
        .get_group(counterfactual)\
        .set_index(['likelihood', 'version'])

    replace = {
        'G': 'ExpTwo',
        'H': 'ExpOne',
        'I': 'ExpThree'
    }

    query = util.load_query()

    fh = open(dest, "w")

    for (lh, version), logk in results.iterrows():
        if lh == "ipe_" + query:
            lh = 'Ipe'
        else:
            lh = "".join([x.capitalize() for x in lh.split("_")])

        cmdname = "BayesFactor{}{}".format(lh, replace[version])
        cmd = r"$\textrm{{\log{{K}}}}={logK:.2f}$".format(**logk)
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
