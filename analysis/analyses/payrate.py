#!/usr/bin/env python

import util
import pandas as pd
from datetime import timedelta

filename = "payrate.csv"


def run(results_path, seed):
    human = util.load_human()
    versions = list(human['all']['version'].unique())
    results = {}
    for version in versions:
        hdata = human['all'].groupby('version').get_group(version)
        starttime = hdata.groupby('pid')['timestamp'].min()
        endtime = hdata.groupby('pid')['timestamp'].max()
        exptime = endtime - starttime
        medtime = timedelta(seconds=float(exptime.median()) / 1e9)
        meantime = timedelta(seconds=float(exptime.mean()) / 1e9)
        if version == "G":
            payrate = (1.0 / (exptime.astype(int) / (1e9 * 60 * 60))).mean()
        elif version == "H":
            payrate = (1.25 / (exptime.astype(int) / (1e9 * 60 * 60))).mean()
        elif version == "I":
            payrate = (0.70 / (exptime.astype(int) / (1e9 * 60 * 60))).mean()
        else:
            raise ValueError("unexpected version: %s" % version)

        results[version] = {
            "median_time": medtime,
            "mean_time": meantime,
            "mean_pay": payrate
        }

    results = pd.DataFrame.from_dict(results).T
    results.index.name = "version"

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
