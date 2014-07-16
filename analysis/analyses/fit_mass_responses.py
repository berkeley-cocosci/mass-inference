#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from path import path
from util import exponentiated_luce_choice as elc


def run(results_path, seed):
    np.random.seed(seed)

    responses = pd.read_csv(path(results_path).dirname().joinpath(
        "mass_responses_by_stimulus.csv"))
    responses = responses\
        .groupby(['version', 'class'])\
        .get_group(('H', 'static'))\
        .set_index(['species', 'kappa0', 'stimulus'])['median']\
        .unstack('species')\
        .drop('prior', level='stimulus')

    empirical = np.asarray(responses['empirical'])
    human = np.asarray(responses['human'])
    ipe = np.asarray(responses['ipe'])

    samps = np.random.rand(10000, 40, human.size) < human
    samps = samps.mean(axis=1)

    empirical_params = np.array([
        curve_fit(elc, empirical, y)[0] for y in samps])
    ipe_params = np.array([
        curve_fit(elc, ipe, y)[0] for y in samps])

    results = pd.DataFrame(
        np.array([
            np.percentile(empirical_params, [2.5, 50, 97.5]),
            np.percentile(ipe_params, [2.5, 50, 97.5])]),
        index=['empirical', 'ipe'],
        columns=['lower', 'median', 'upper'])
    results.index.name = 'model'

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
