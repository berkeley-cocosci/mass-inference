#!/usr/bin/env python

import pandas as pd
import numpy as np
import util

filename = "fall_response_corrs.csv"
texname = "fall_response_corrs.tex"


def run(data, results_path, version, seed):
    np.random.seed(seed)

    means = pd\
        .read_csv(results_path.joinpath(version, "fall_responses.csv"))\
        .set_index(['block', 'species', 'kappa0', 'stimulus'])['median']

    results = {}
    for block, df in means.groupby(level='block'):
        m = df.unstack(['species', 'kappa0'])

        # human vs human
        x = m[('human', -1.0)]
        y = m[('human', 1.0)]
        results[(block, 'Human', 'Human')] = util.bootcorr(x, y)

        # mass-sensitive ipe vs human
        x = pd.concat([m[('model', -1.0)], m[('model', 1.0)]])
        y = pd.concat([m[('human', -1.0)], m[('human', 1.0)]])
        results[(block, 'ModelS', 'Human')] = util.bootcorr(x, y)

        # mass-insensitive ipe vs human
        x = pd.concat([m[('model', 0.0)], m[('model', 0.0)]])
        y = pd.concat([m[('human', -1.0)], m[('human', 1.0)]])
        results[(block, 'ModelIS', 'Human')] = util.bootcorr(x, y)

    results = pd.DataFrame.from_dict(results, orient='index')
    results.index = pd.MultiIndex.from_tuples(
        results.index,
        names=['block', 'X', 'Y'])

    pth = results_path.joinpath(version, filename)
    results.to_csv(pth)

    with open(results_path.joinpath(version, texname), "w") as fh:
        fh.write("%% AUTOMATICALLY GENERATED -- DO NOT EDIT!\n")
        for (block, X, Y), stats in results.iterrows():
            cmd = util.newcommand(
                "%sv%sFallCorr%s" % (X, Y, block),
                util.latex_pearson.format(**dict(stats)))
            fh.write(cmd)

    return pth


if __name__ == "__main__":
    util.run_analysis(run)
