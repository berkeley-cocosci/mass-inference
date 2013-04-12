#!/usr/bin/python
"""Parses raw experiment csv files into numpy arrays"""

import numpy as np
import sqlite3 as sql
import os
import pickle

exp_ver = "F"

# render_path = "../../stimuli/render"
meta_dir = "../../stimuli/meta"
data_dir = "../../data/human/raw_data-%s" % exp_ver
data_out_dir = "../../data/human/processed_data"


def convert_stim_name(name):
    infopath = os.path.join(meta_dir, "%s-conversion.pkl" % exp_ver)
    with open(infopath, "r") as fh:
        # lines = [x for x in fh.read().strip().split("\n") if x != ""]
        conv = pickle.load(fh)
    # newnames = [x.split(",")[0] for x in lines]
    # oldnames = [x.split(",")[1] for x in lines]
    # idx = newnames.index(name)
    oldname = conv[name]
    return oldname


def get_completed(data_db):
    conn = sql.connect(data_db)
    with conn:
        cur = conn.cursor()
        cur.execute("SELECT pid,condition,completion_code FROM Participants")
        vals = cur.fetchall()
    good = [(x[0], str(x[1])) for x in vals if x[2] is not None]
    return good

data = get_completed(os.path.join(data_dir, "data.db"))
pids, conds = zip(*data)
pids = ["%03d" % p for p in pids]

DTYPE = np.dtype(
    [("index", np.int),
     ("trial", np.int),
     ("stimulus", np.dtype('<S50')),
     ("response", np.dtype('<S10')),
     ("time", np.float),
     ("angle", np.int),
     ("ttype", np.dtype('<S15'))])

for idx in xrange(len(pids)):
    pid = pids[idx]
    cond = conds[idx]

    datafile = "%s.csv" % pid
    triallist = "%s_trials.json" % pid

    if not os.path.exists(os.path.join(data_dir, datafile)):
        continue

    print "Processing '%s'..." % pid
    with open(os.path.join(data_dir, datafile), "r") as fh:
        lines = [x for x in fh.read().strip().split("\n") if x != ""]

    fields = lines.pop(0).split(",")
    assert fields == list(DTYPE.names)

    training = []
    experiment = []
    posttest = []
    queries = []
    curr = training

    for line in lines:
        vals = line.split(",")

        if vals[1] == "finished training":
            curr = experiment
        elif vals[1] == "finished experiment":
            curr = posttest
        elif vals[1] == "finished posttest":
            pass
        elif vals[1] == "query ratio":
            index = vals[0]
            newvals = []
            for vidx, val in enumerate(vals):
                if DTYPE.names[vidx] == "trial":
                    newvals.append(index)
                elif DTYPE.names[vidx] == "stimulus":
                    newvals.append("query ratio")
                elif DTYPE.names[vidx] == "angle":
                    newvals.append(0)
                elif DTYPE.names[vidx] == "ttype":
                    newvals.append("query")
                else:
                    newvals.append(val)
            queries.append(tuple(newvals))

        else:
            newvals = []
            for vidx, val in enumerate(vals):
                if DTYPE.names[vidx] == "stimulus":
                    newval = convert_stim_name(val)
                else:
                    newval = val
                newvals.append(newval)
            curr.append(tuple(newvals))

    arrs = {
        "training":  np.array(training, dtype=DTYPE),
        "experiment": np.array(experiment, dtype=DTYPE),
        "posttest": np.array(posttest, dtype=DTYPE)
        }

    if cond.split("-")[1] in ("fb", "vfb"):
        arrs['queries'] = np.array(queries, dtype=DTYPE)

    if not os.path.exists(data_out_dir):
        os.mkdir(data_out_dir)

    datapath = os.path.join(data_out_dir, "%s~%s.npz" % (pid, cond))
    np.savez(datapath, **arrs)

    print "Saved to '%s'." % datapath
