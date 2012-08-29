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

reload(mo)
reload(lat)

rawhuman, rawhstim, raworder, msamp, params = lat.load('stability')
sigmas, phis, kappas = params
ratios = np.round(10 ** kappas, decimals=1)

human, stimuli, sort, model = lat.order_by_trial(
    rawhuman, rawhstim, raworder, msamp)

predicates = list(model.dtype.names)
predicates.remove('stability_pfell')
#predicates.remove('direction')
predicates.remove('stability_nfell')
predicates.remove('radius')

# variables
n_trial      = stimuli.shape[1]
n_kappas     = len(kappas)
#n_outcomes   = (11, 8)
#n_outcomes   = (5, 8)
#n_outcomes   = (11,)
n_outcomes   = (16,)
n_predicate  = len(predicates)

# samples from the IME
ime_samps = model[0, 1].copy()
#ime_samps = model[0, 0][:, :, [0]].copy()
# true outcomes for each mass ratio
truth = model[0, 0, :, :, 0].copy()


ratio = 10
kidx  = list(ratios).index(ratio)
B     = 10   # mixing parameter

n_trial     = stimuli.shape[1]
n_kappas    = len(kappas)
n_part      = 100
n_responses = 7

# samples from the IME
p_outcomes = mo.IME(ime_samps, n_outcomes, predicates)

import kde

#from kde import gen_direction_edges
#edges, binsize, offset = gen_direction_edges(16)
fig, axes = plt.subplots(3, 4)#, subplot_kw=dict(polar=True))

# def hist(ax, e, z, s, t, title=""):
#     plt.axes(ax)
#     ax.cla()
#     ax.bar(e, z, width=e[1]-e[0], bottom=0.3)
#     ax.plot(s['direction'], s['radius'], 'ro')
#     ax.plot([t['direction']]*2, [0, t['radius']], 'g-', linewidth=5)
#     ax.set_ylim(0,0.5)
#     ax.set_title(title)
#     plt.box(False)
#     plt.yticks([], [])
#     plt.draw()

t = ime_samps['direction']
r = ime_samps['radius']

tt = truth['direction']
tr = truth['radius']

x = np.cos(t)*r
y = np.sin(t)*r

tx = np.cos(tt)*tr
ty = np.sin(tt)*tr

# x = np.empty((1, 1, 48))
# x[:, :, :24] = 0.2205
# x[:, :, 24:] = -0.2205
# y = np.empty((1, 1, 48))
# y[:, :, :24] = -0.2205
# y[:, :, 24:] = 0.2205

# tx = np.zeros((1, 1))
# ty = np.zeros((1, 1))

data = np.concatenate([x[..., None], y[..., None]], axis=-1)

n = (20, 20)
edges, binsize = kde.gen_xy_edges(n)
mids = (edges[:, 1:] + edges[:, :-1]) / 2.

sclx = lambda x: (((x - edges[0,0]) / (edges[0,-1]-edges[0,0])) * (edges.shape[1]-1)) - 0.5
scly = lambda y: (((y - edges[1,0]) / (edges[1,-1]-edges[1,0])) * (edges.shape[1]-1)) - 0.5

if os.path.exists("xy_kde.npy"):
    bx = np.load("xy_kde.npy")
else:
    bx = np.empty(data.shape[:-2] + n)
    for i in xrange(data.shape[0]):
        print "%d / %d" % (i, data.shape[0])
        bx[i] = kde.xy_kde(data[i], n, h=0.2, s=-.35, t=.35)
    np.save("xy_kde.npy", bx)

for i in xrange(data.shape[0]):
    for j in xrange(data.shape[1]):
        title = "r=%.1f" % ratios[j]
        ax = axes.ravel()[j]
        plt.axes(ax)
        ax.cla()
        ax.imshow(bx[i,j].T, interpolation='nearest', vmin=0, vmax=1)
        ax.plot(sclx(x[i,j]), scly(y[i,j]), 'ro')
        ax.plot(sclx(tx[i,j]), scly(ty[i,j]), 'yo')
        ax.set_xticks(np.arange(len(edges[0]))-0.5)
        ax.set_xticklabels(edges[0])
        ax.set_yticks((np.arange(len(edges[0]))-0.5))
        ax.set_yticklabels(edges[1])
        ax.set_title(title)
        plt.draw()
        # hist(axes.ravel()[j],
        #      edges[:-1],
        #      np.exp(p_outcomes[i,j]),
        #      ime_samps[i,j],
        #      truth[i,j],
        #      title=title)
        print title#, model_subjects[j,i]
    pdb.set_trace()

