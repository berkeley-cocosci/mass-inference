#!/usr/bin/env python

import sys
import util


def run(results_path, seed):
    human = util.load_human()
    order = human['all']\
        .set_index(['mode', 'trial', 'pid'])['stimulus']\
        .unstack('pid')

    order.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
