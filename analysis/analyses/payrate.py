#!/usr/bin/env python

import util
import pandas as pd
from datetime import timedelta

filename = "payrate.csv"


def run(data, results_path, version, seed):
    hdata = data['human']['all']
    starttime = hdata.groupby('pid').apply(
        lambda x: x.sort('timestamp')['timestamp'].min())
    endtime = hdata.groupby('pid').apply(
        lambda x: x.sort('timestamp')['timestamp'].max())
    exptime = endtime - starttime
    medtime = timedelta(seconds=float(exptime.median()) / 1e9)
    meantime = timedelta(seconds=float(exptime.mean()) / 1e9)
    if version == "G":
        payrate = (1.0 / (exptime.astype(int) / (1e9 * 60 * 60))).mean()
    elif version == "H":
        payrate = (1.25 / (exptime.astype(int) / (1e9 * 60 * 60))).mean()
    else:
        raise ValueError("unexpected version: %s" % version)

    results = pd.Series(
        [medtime, meantime, payrate],
        index=['median_time', 'mean_time', 'mean_pay'])
    results = results.to_frame().reset_index()
    results.columns = ['key', 'value']
    results = results.set_index('key')

    pth = results_path.joinpath(version, filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
