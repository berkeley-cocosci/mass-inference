#!/usr/bin/env python

"""
Computes the squared error between model and human judgments on "will it fall?"
for each stimulus for all the different parameter combinations of sigma/phi.
Produces a csv file with the following columns:

    stimulus (string)
        stimulus name
    kappa (float)
        log mass ratio
    sigma (float)
        perceptual uncertainty
    phi (float)
        force uncertainty
    sqerr (float)
        squared error between people and model

"""

__depends__ = ["human_fall_responses.csv", "model_fall_responses.csv"]

import util
import pandas as pd
import numpy as np
from path import path


def run(results_path):
    human = pd.read_csv(path(results_path).dirname().joinpath(
        "fall_responses.csv"))

    human = human\
        .set_index(['version', 'block', 'species', 'stimulus', 'kappa0'])\
        .ix[('GH', 'B', 'human')]['median']\
        .sortlevel()

    ipe = util.load_model()[0]['B']
    model = ipe.P_fall_mean_all[[-1.0, 1.0]].stack()

    results = []
    for (sigma, phi), model_df in model.groupby(level=['sigma', 'phi']):
        x = model_df.reset_index(['sigma', 'phi'], drop=True)
        err = (x - human) ** 2
        err.index = model_df.index
        results.append(err)

    results = pd\
        .concat(results)\
        .reset_index(['sigma', 'phi'])\
        .rename(columns={0: 'sqerr'})

    results.to_csv(results_path)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.results_path)
