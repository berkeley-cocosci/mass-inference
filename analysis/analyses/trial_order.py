#!/usr/bin/env python

import util

filename = "trial_order.csv"


def run(data, results_path, seed):
    order = data['human']['all']\
        .set_index(['mode', 'trial', 'pid'])['stimulus']\
        .unstack('pid')

    pth = results_path.joinpath(filename)
    order.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
