#!/usr/bin/env python

import util

filename = "trial_order.csv"


def run(results_path, seed):
    human = util.load_human()
    order = human['all']\
        .set_index(['mode', 'trial', 'pid'])['stimulus']\
        .unstack('pid')

    pth = results_path.joinpath(filename)
    order.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
