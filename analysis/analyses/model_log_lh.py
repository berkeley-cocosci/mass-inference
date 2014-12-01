#!/usr/bin/env python

"""
Computes the log likelihood of every response for every participant under each
of the different models. Produces a csv file with the following columns:

    likelihood (string)
        the likelihood (e.g. empirical, ipe)
    counterfactual (bool)
        whether the counterfactual likelihood was used
    model (string)
        the model type (e.g. static, learning)
    fitted (bool)
        whether the model was fitted to human data or not
    version (string)
        the experiment version
    pid (string)
        the participant id
    num_mass_trials (int)
        how many trials the participant responded to "which is heavier?"
    trial (int)
        the trial number
    llh (float)
        the log likelihood of the participant's response

"""

__depends__ = ["human", "single_model_belief.csv"]

import os
import util
import pandas as pd
import numpy as np



def compute_log_lh(model_df, human_df):
    m = np.asarray(model_df)
    h = np.asarray(human_df)
    lh = pd.DataFrame(
        np.log(((m * h) + ((1 - m) * (1 - h)))),
        index=model_df.index,
        columns=model_df.columns)
    lh.name = model_df.name
    return lh


def run(dest, data_path, results_path):
    all_human = util.load_human(data_path)['C']
    human = all_human\
        .set_index(['version', 'pid', 'trial'])['mass? response']\
        .unstack('trial')\
        .sortlevel()
    # convert from -1,1 to 0,1
    human = (human + 1) / 2.0

    cols = ['likelihood', 'counterfactual', 'model', 'fitted']
    model = pd\
        .read_csv(os.path.join(results_path, "single_model_belief.csv"))\
        .set_index(cols + ['version', 'pid', 'trial'])['p']\
        .unstack('trial')\
        .sortlevel()

    llh = model\
        .groupby(level=cols)\
        .apply(compute_log_lh, human)

    results = pd.melt(
        llh.reset_index(),
        id_vars=cols + ['version', 'pid'],
        var_name='trial',
        value_name='llh')

    results = pd.merge(
        results.dropna(),
        all_human[['pid', 'num_mass_trials']].drop_duplicates())

    results = results.set_index(cols).sortlevel()
    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.dest, args.data_path, args.results_path)
