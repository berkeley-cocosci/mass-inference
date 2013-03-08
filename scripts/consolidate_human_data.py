"""Consolidates numpy arrays of individual participants into a single
array"""

import os
import numpy as np

ttypes = ["training", "experiment", "queries", "posttest"]
data_path = "../data/human/processed_data"
data_out_path = "../data/human/consolidated_data/%s_data~%s.npz"

for ttype in ttypes:

    # list of data files to consolidate
    files = sorted([x for x in os.listdir(data_path) if x.endswith(".npz")])

    # load the data
    dfiles = []
    subjs = []
    for f in files:
        try:
            subjs.append(np.load(os.path.join(data_path, f))[ttype])
        except KeyError:
            pass
        else:
            dfiles.append(os.path.splitext(f)[0])

    allpids, conds = zip(*[x.split("~") for x in dfiles])
    allconds = [str(x) for x in np.unique(conds)]
    nconds = len(allconds)
    
    # the fields we want to store
    dtype = np.dtype([
        ("response", 'f8'),
        ("trial", 'i8'),
        ("time", 'f8'),
        ("angle", 'f8'),
        ])
    ndata = len(dtype)

    for cond in allconds:

        if cond.endswith("-cb0"):
            counterbalance = False
        elif cond.endswith("-cb1"):
            counterbalance = True
        else:
            counterbalance = False

        sidxs = np.nonzero(np.array(conds) == cond)[0]
        nsubj = len(sidxs)
        pids = np.array(allpids)[sidxs]
        
        # figure out stimuli
        stims = sorted([str(x) for x in subjs[sidxs[0]]['stimulus']])
        nstim = len(stims)

        # orders = []
            #  = np.array([
            # np.argsort(subjs[sidxs][i]['stimulus']) 
            # for i in xrange(nsubj)])
    
        # allocate data array
        hdata = np.empty([nstim, nsubj], dtype=dtype)
        
        # load in the data
        for idx, sidx in enumerate(sidxs):
            hsort = np.argsort(subjs[sidx]['stimulus'])

            for didx in xrange(ndata):

                dname = dtype.names[didx]
                data = subjs[sidx][dname][hsort]

                if dname == "response":
                    if ttype == "queries":
                        ratio = float(cond.split("-")[2])
                        newdata = np.zeros(data.shape, dtype="f8")
                        if not counterbalance:
                            # counterbalance = 0 --> r=yellow:red 

                            # if they say yellow is heavier, then they
                            # think the ratio is 10:1

                            # if they say red is heavier, then they
                            # think the ratios is 1:10
                            
                            newdata[data == "red"] = 0.1
                            newdata[data == "yellow"] = 10.0
                        else:
                            # counterbalance = 1 --> r=red:yellow

                            # if they say yellow is heavier, then they
                            # think the ratio is 1:10

                            # if they say red is heavier, then they
                            # think the ratios is 10:1
                            
                            newdata[data == "red"] = 10.0
                            newdata[data == "yellow"] = 0.1
                        assert (newdata != 0).all()
                        data = newdata
                        
                    else:    
                        # convert from yes/no to True/False
                        newdata = np.empty(data.shape, dtype=bool)
                        newdata[data == "yes"] = True
                        newdata[data == "no"] = False
                        data = newdata
                # elif dname == "trial":
                #     # zero-index the trials
                #     data -= np.min(data)

                try:
                    hdata[:, idx][dname] = data
                except:
                    print pids[idx], ttype
                    raise

        out_path = data_out_path % (ttype, cond)
        print "Saving '%s'..." % out_path
        np.savez(out_path,
                 data=hdata,
                 stims=stims,
                 pids=pids)
