#!/usr/bin/env python

import util
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

filename = "fit_mass_responses.csv"


def run(results_path, seed):
    np.random.seed(seed)

    responses = pd\
        .read_csv("results/mass_responses_by_stimulus.csv")\
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
        curve_fit(util.sigmoid, empirical, y)[0] for y in samps])
    ipe_params = np.array([
        curve_fit(util.sigmoid, ipe, y)[0] for y in samps])

    results = pd.DataFrame(
        np.array([
            np.percentile(empirical_params, [2.5, 50, 97.5]),
            np.percentile(ipe_params, [2.5, 50, 97.5])]),
        index=['empirical', 'ipe'],
        columns=['lower', 'median', 'upper'])
    results.index.name = 'model'

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)



    # print popt_x, popt_m

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True, sharey=True)
    # ax1.plot(x, y, 'ko')
    # ax1.plot(np.linspace(0, 1, 100), sigmoid(np.linspace(0, 1, 100), *popt_x), 'r-')
    # ax2.plot(sigmoid(x, *popt_x), y, 'ko')
    # ax3.plot(m, y, 'ko')
    # ax3.plot(np.linspace(0, 1, 100), sigmoid(np.linspace(0, 1, 100), *popt_x), 'r-')
    # ax4.plot(sigmoid(m, *popt_x), y, 'ko')

    # fig.set_figwidth(15)
    # fig.set_figheight(3.5)
