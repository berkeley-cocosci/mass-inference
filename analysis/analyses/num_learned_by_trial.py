#!/usr/bin/env python

import util
import pandas as pd
import scipy.stats

filename = "num_learned_by_trial.csv"


def run(results_path, seed):

    sp = pd\
        .read_csv(results_path.joinpath('switchpoint.csv'))

    sp_all = sp.copy()
    sp_all['kappa0'] = 'all'
    sp = pd.concat([sp, sp_all])\
        .set_index(['version', 'kappa0', 'pid'])

    num_mass_trials = (~sp.isnull()).sum(axis=1)
    sp['num_mass_trials'] = num_mass_trials
    sp = sp.set_index('num_mass_trials', append=True)
    sp.columns.name = 'trial'

    results = sp\
        .stack()\
        .groupby(level=['version', 'kappa0', 'num_mass_trials', 'trial'])\
        .apply(util.beta)\
        .unstack(-1)

    p = pd\
        .read_csv(results_path.joinpath('mass_accuracy.csv'))\
        .set_index(['species', 'class', 'version', 'kappa0'])\
        .ix[('human', 'static', 'H', 'all')]['median']

    num = results['median']\
        .groupby(level='kappa0')\
        .get_group('all')

    num[:] = 1
    num = num.unstack(-1).T.cumsum()
    num = num.max() - num + 1

    chance = scipy.stats.binom.pmf(num, num, p)
    chance = pd.DataFrame(chance, index=num.index, columns=num.columns)
    chance = pd.DataFrame(chance.T.stack(), columns=['median']).reset_index()
    chance['kappa0'] = 'chance'
    chance = chance.set_index(
        ['version', 'kappa0', 'num_mass_trials', 'trial'])

    results = pd.concat([results, chance])

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
