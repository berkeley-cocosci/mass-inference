import collections
import matplotlib.cm as cm
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

from matplotlib import rc
from sklearn import linear_model

from cogphysics.lib.corr import xcorr

import model_observer as mo
import learning_analysis_tools as lat

normalize = rvs.util.normalize
weightedSample = rvs.util.weightedSample

######################################################################
## Load and process data

rawhuman, rawhstim, raworder, rawtruth, rawipe, kappas = lat.load('stability')
ratios = np.round(10 ** kappas, decimals=1)

human, stimuli, sort, truth, ipe = lat.order_by_trial(
    rawhuman, rawhstim, raworder, truth, ipe)

