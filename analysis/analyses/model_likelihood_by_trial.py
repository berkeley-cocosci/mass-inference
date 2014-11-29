#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path
from IPython.parallel import Client, require


@require('numpy', 'pandas')
def task(args):
    key, participants, orders, old_store_pth = args

    pd = pandas
    np = numpy

    old_store = pd.HDFStore(old_store_pth, mode='r')
    llh = old_store[key]\
        .reset_index('kappa0')
    llh['kappa0'] = llh['kappa0'].apply(str)
    llh = llh\
        .set_index('kappa0', append=True)\
        .reorder_levels(['kappa0', 'stimulus'])
    hyps = llh.columns

    results = []
    for (version, kappa0), pids in participants.groupby(level=['version', 'kappa0']):
        df = np.asarray(llh.ix[str(kappa0)])

        for pid in pids:
            order, i = orders[pid]
            model = pd.DataFrame(df[i], index=order['stimulus'], columns=hyps)
            model.index.name = 'stimulus'
            model.columns.name = 'hypothesis'
            model['trial'] = np.asarray(order['trial'])
            model = model.set_index('trial', append=True).stack()
            model.name = 'logp'
            model = model.reset_index()
            model['kappa0'] = kappa0
            model['pid'] = pid
            model['version'] = version
            model = model.set_index([
                'version', 'kappa0', 'pid', 'trial', 'hypothesis'])
            results.append(model)

    return key, pd.concat(results)


def run(results_path, seed):
    np.random.seed(seed)
    data = util.load_human()

    old_store_pth = path(results_path).dirname().joinpath(
        'model_likelihood.h5').abspath()
    old_store = pd.HDFStore(old_store_pth, mode='r')
    store = pd.HDFStore(results_path, mode='w')

    pth = path(results_path).dirname().joinpath("trial_order.csv")
    trials = pd.read_csv(pth)\
               .set_index(['mode', 'trial', 'pid'])['stimulus']\
               .unstack('pid')\
               .ix['experimentC']

    human = data['C']
    participants = human[['version', 'kappa0', 'pid']]\
        .drop_duplicates()\
        .sort(['version', 'kappa0', 'pid'])\
        .set_index(['version', 'kappa0'])['pid']

    orders = {}
    for pid in participants:
        order = trials[pid].dropna()
        order.name = 'stimulus'
        order = order.reset_index()
        i = np.argsort(order.sort('stimulus')['trial'])
        orders[pid] = (order, i)

    rc = Client()
    lview = rc.load_balanced_view()
    results = []

    for key in old_store.keys():
        if key.split('/')[-1] == 'param_ref':
            store.append(key, old_store[key])
            continue

        print "starting:", key
        result = lview.apply(task, [key, participants, orders, old_store_pth])
        results.append(result)

    while len(results) > 0:
        result = results.pop(0)
        if not result.ready():
            results.append(result)
            continue

        key, model = result.get()
        print "finished:", key
        store.append(key, model)

    store.close()
    old_store.close()

if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
