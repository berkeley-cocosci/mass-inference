#!/usr/bin/env python

__depends__ = ["fit_sigmoids.csv"]

import os
import util
import pandas as pd


def run(dest, results_path, query):
    results = pd\
        .read_csv(os.path.join(results_path, 'fit_sigmoids.csv'))\
        .query('(likelihood == "empirical") | (likelihood == "ipe_{}")'.format(query))\
        .set_index(['likelihood', 'counterfactual', 'random'])
    latex_beta = util.load_config()["latex"]["beta"]

    fh = open(dest, "w")
    for (lh, cf, random), row in results.iterrows():
        if lh == "ipe_" + query:
            lh = 'Ipe'
        else:
            lh = lh.capitalize()
        if cf:
            cf = "CF"
        else:
            cf = ""
        if random:
            random = "Random"
        else:
            random = ""

        cmdname = "SigmoidCoef{}{}{}".format(lh, cf, random)
        value = latex_beta.format(**row)
        cmd = util.newcommand(cmdname, value)
        fh.write(cmd)

    fh.close()


if __name__ == "__main__":
    config = util.load_config()
    parser = util.default_argparser(locals())
    parser.add_argument(
        '--query',
        default=config['analysis']['query'],
        help='which query for the ipe to use')
    args = parser.parse_args()
    run(args.to, args.results_path, args.query)
