#!/usr/bin/env python

import pandas as pd
from mass import DATA_PATH
from snippets import datapackage as dpkg

data_G_path = DATA_PATH.joinpath("human", "mass_inference-G.dpkg")
data_H_path = DATA_PATH.joinpath("human", "mass_inference-H.dpkg")
data_I_path = DATA_PATH.joinpath("human", "mass_inference-I.dpkg")
data_path = DATA_PATH.joinpath("human", "mass_inference-merged.dpkg")

dp_G = dpkg.DataPackage.load(data_G_path)
dp_H = dpkg.DataPackage.load(data_H_path)
dp_I = dpkg.DataPackage.load(data_I_path)

dp = dpkg.DataPackage(name=data_path.name, licenses=['odc-by'])
dp['version'] = '1.0.0'
dp.add_contributor("Jessica B. Hamrick", "jhamrick@berkeley.edu")
dp.add_contributor("Thomas L. Griffiths", "tom_griffiths@berkeley.edu")
dp.add_contributor("Peter W. Battaglia", "pbatt@mit.edu")
dp.add_contributor("Joshua B. Tenenbaum", "jbt@mit.edu")

# add event data, and save it as csv
r = dpkg.Resource(
    name="events-G.csv", fmt="csv",
    pth="./events-G.csv",
    data=dp_G.load_resource("events.csv"))
r['mediaformat'] = 'text/csv'
dp.add_resource(r)

r = dpkg.Resource(
    name="events-H.csv", fmt="csv",
    pth="./events-H.csv",
    data=dp_H.load_resource("events.csv"))
r['mediaformat'] = 'text/csv'
dp.add_resource(r)

r = dpkg.Resource(
    name="events-I.csv", fmt="csv",
    pth="./events-I.csv",
    data=dp_I.load_resource("events.csv"))
r['mediaformat'] = 'text/csv'
dp.add_resource(r)

# add participant info, and save it as csv
r = dpkg.Resource(
    name="participants-G.csv", fmt="csv",
    pth="./participants-G.csv",
    data=dp_G.load_resource("participants.csv"))
r['mediaformat'] = 'text/csv'
dp.add_resource(r)

r = dpkg.Resource(
    name="participants-H.csv", fmt="csv",
    pth="./participants-H.csv",
    data=dp_H.load_resource("participants.csv"))
r['mediaformat'] = 'text/csv'
dp.add_resource(r)

r = dpkg.Resource(
    name="participants-I.csv", fmt="csv",
    pth="./participants-I.csv",
    data=dp_I.load_resource("participants.csv"))
r['mediaformat'] = 'text/csv'
dp.add_resource(r)

# add metadata, and save it inline as json
r = dpkg.Resource(
    name="metadata-G", fmt="json",
    data=dp_G.load_resource("metadata"))
r['mediaformat'] = 'application/json'
dp.add_resource(r)

r = dpkg.Resource(
    name="metadata-H", fmt="json",
    data=dp_H.load_resource("metadata"))
r['mediaformat'] = 'application/json'
dp.add_resource(r)

r = dpkg.Resource(
    name="metadata-I", fmt="json",
    data=dp_I.load_resource("metadata"))
r['mediaformat'] = 'application/json'
dp.add_resource(r)

# add experiment data, and save it as csv
exp_G = dp_G.load_resource("experiment.csv")
exp_H = dp_H.load_resource("experiment.csv")
exp_I = dp_I.load_resource("experiment.csv")

exp_G['version'] = 'G'
exp_H['version'] = 'H'
exp_I['version'] = 'I'

# TODO: fix this hack
exp_I.loc[exp_I['mode'] == 'experimentB', 'mode'] = 'experimentC'

exp = pd.concat([exp_G, exp_H, exp_I])

r = dpkg.Resource(
    name="experiment.csv", fmt="csv",
    pth="./experiment.csv", data=exp)
r['mediaformat'] = 'text/csv'
dp.add_resource(r)

# save the datapackage
dp.save(data_path.dirname())
print("Saved to '%s'" % data_path.relpath())
