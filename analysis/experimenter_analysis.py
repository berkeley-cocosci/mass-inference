import collections
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np
import pdb
import pickle
import scipy.stats
import os
import time

import cogphysics
import cogphysics.lib.circ as circ
import cogphysics.lib.nplib as npl
import cogphysics.lib.rvs as rvs
import cogphysics.tower.analysis_tools as tat

from cogphysics.lib.corr import xcorr

import model_observer as mo
import learning_analysis_tools as lat

normalize = rvs.util.normalize
weightedSample = rvs.util.weightedSample

cmap = lat.make_cmap("lh", (0, 0, 0), (.5, .5, .5), (1, 0, 0))

######################################################################
## Load and process data

rawhuman, rawhstim, raworder, rawtruth, rawipe, kappas = lat.load('stability')
ratios = 10 ** kappas
ratios[kappas < 0] = np.round(ratios[kappas < 0], decimals=2)
ratios[kappas >= 0] = np.round(ratios[kappas >= 0], decimals=1)

human, stimuli, sort, truth, ipe = lat.order_by_trial(
    rawhuman, rawhstim, raworder, rawtruth, rawipe)
truth = truth[0]
ipe = ipe[0]
 
# variables
n_trial      = stimuli.shape[1]
n_kappas     = len(kappas)

######################################################################
# Model observer for each true mass ratio

def make_data(nthresh0, nthresh, nsamps):
    ipe_samps = np.concatenate([
        ((ipe['nfellA'] + ipe['nfellB']) > nthresh).astype('int')[..., None],
        ], axis=-1)[..., :nsamps, :]
    feedback = np.concatenate([
        ((truth['nfellA'] + truth['nfellB']) > nthresh0).astype('int'),
        ], axis=-1)
    return feedback, ipe_samps

vals = {}
def mem_model_observer(nthresh0, nthresh, nsamps, smooth, decay):
    dparams = (nthresh0, nthresh, nsamps)
    params = (smooth, decay)
    if dparams not in vals:
        vals[dparams] = {}
    if params not in vals[dparams]:
        feedback, ipe_samps = make_data(*dparams)
        out = mo.ModelObserver(
            ipe_samps,
            feedback[:, None],
            smooth=smooth,
            decay=decay)
        vals[dparams][params] = out
    return vals[dparams][params]

