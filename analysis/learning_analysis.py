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

from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab

normalize = rvs.util.normalize
weightedSample = rvs.util.weightedSample

def make_cmap(name, c1, c2, c3):

    colors = {
        'red'   : (
            (0.0, c1[0], c1[0]),
            (0.50, c2[0], c2[0]),
            (1.0, c3[0], c3[0]),
            ),
        'green' : (
            (0.0, c1[1], c1[1]),
            (0.50, c2[1], c2[1]),
            (1.0, c3[1], c3[1]),
            ),
        'blue'  : (
            (0.0, c1[2], c1[2]),
            (0.50, c2[2], c2[2]),
            (1.0, c3[2], c3[2])
            )
        }
    
    cmap = matplotlib.colors.LinearSegmentedColormap(name, colors, 1024)
    return cmap

######################################################################
## Load and process data

rawhuman, rawhstim, raworder, rawtruth, rawipe, kappas = lat.load('stability')
ratios = np.round(10 ** kappas, decimals=2)

human, stimuli, sort, truth, ipe = lat.order_by_trial(
    rawhuman, rawhstim, raworder, rawtruth, rawipe)
truth = truth[0]
ipe = ipe[0]

# variables
n_trial      = stimuli.shape[1]
n_kappas     = len(kappas)

## P(F | S, k)
ipe_samps = np.concatenate([
    #ipe['nfellA'][..., None],
    #ipe['nfellB'][..., None],
    # ipe['comdiff']['x'][..., None],
    # ipe['comdiff']['y'][..., None],
    # ipe['comdiff']['z'][..., None],
    ((ipe['nfellA'] + ipe['nfellB']) > 4).astype('int')[..., None],
    ], axis=-1)
feedback = np.concatenate([
    #truth['nfellA'],
    #truth['nfellB'],
    # truth['comdiff']['x'],
    # truth['comdiff']['y'],
    # truth['comdiff']['z'],
    ((truth['nfellA'] + truth['nfellB']) > 4).astype('int'),
    ], axis=-1)

######################################################################
# Model observers for each true mass ratio

cmap = make_cmap("lh", (0, 0, 0), (.5, .5, .5), (1, 0, 0))
vals = {}

def run_and_plot(decay, h, u, fignum):
    # memoize
    params = (decay, h, u)
    if params not in vals:
        out = mo.ModelObserver(
            ipe_samps,
            feedback[:, None],
            h=h,
            u=u,
            decay=decay)
        vals[params] = out

    model_lh, model_joint, model_theta = vals[params]

    # plot it
    plt.figure(fignum)
    plt.clf()
    plt.suptitle("Posterior P(kappa|F)\ndecay=%.2f, smooth=%.2f, mix=%.2f" % (decay, h, u))
    plt.subplots_adjust(wspace=0.3, hspace=0.2, left=0.08, right=0.95, top=0.92, bottom=0.05)
    for kidx, ratio in enumerate(ratios):
        subjname = "Model Subj. r=%.2f" % ratios[kidx]
        lat.plot_theta(
            6, 5, kidx+1,
            np.exp(model_theta[kidx]),
            subjname,
            exp=1.234,
            ratios=ratios,
            cmap=cmap)
        plt.ylabel('')
        plt.plot(np.arange(n_trial), np.ones(n_trial)*kidx, 'y--', linewidth=3)
        plt.draw()

    plt.colorbar()

    return model_lh, model_joint, model_theta


lh, jnt, th = run_and_plot(
    decay=1.0,   # weight decay rate
    h=0.5,       # gaussian kernel smoothing parameter
    u=0.0,       # mixture parameter for uniform component
    fignum=10
)

lh, jnt, th = run_and_plot(
    decay=1.0,   # weight decay rate
    h=0.5,       # gaussian kernel smoothing parameter
    u=0.1,       # mixture parameter for uniform component
    fignum=11
)

lh, jnt, th = run_and_plot(
    decay=0.99,  # weight decay rate
    h=0.5,       # gaussian kernel smoothing parameter
    u=0.1,       # mixture parameter for uniform component
    fignum=12
)


d0 = scipy.stats.nanmean(np.sqrt(np.sum(ipe_samps**2, axis=-1)), axis=-1)
h0 = np.mean(human, axis=0)

for i in xrange(n_kappas):
    print ratios[i], np.corrcoef(d0[:, i], h0)[0,1]

plt.clf()
plt.plot(
    np.mean(human, axis=0),
    np.mean(d0, axis=-2)[:, -1], 'bo')


transform = lambda x: 1 - np.exp(-x)
invtransform = lambda y: -np.log(1 - y)

d0 = np.sqrt(np.sum(ipe_samps**2, axis=-1)).ravel()
d0 = np.sort(transform(d0[~np.isnan(d0)]))
#d0 = np.sort(np.exp(-np.sqrt(np.sum(feedback**2, axis=-1)).ravel()))
a7 = np.array_split(d0, 7)
edges = np.array(
    [transform(0)] +
    [(a7[i][-1] + a7[i+1][0]) / 2.
     for i in xrange(6)] +
    [transform(3)])
edges = (edges - edges[0]) / (edges[-1] - edges[0])
edges = invtransform(edges)




# mthresh = 0.1
# #d = d0[d0 > mthresh]# - mthresh
# d = d0.copy()

# N = d.size
# meand = np.mean(d)
# logd = np.log(d)
# sumlogd = np.sum(logd)
# sumd = np.sum(d)
# def lh(k):
#     #if k <= 0:
#     #    return -np.inf
#     theta = meand / k
#     logtheta = np.log(theta)
#     gammaln = scipy.special.gammaln
#     l = ((k - 1) * sumlogd) - (sumd / theta) - (N*k*logtheta) - (N*gammaln(k))
#     try:
#         if np.isnan(l):
#             l = -np.inf
#         print "l(%.2f, %.2f) = %f" % (k, theta, l)
#     except:
#         pass
#     return l

# def dlh(k):
#     dx = 0.006
#     x = np.array([k-dx, k+dx])
#     y = lh(x)
#     dy = (y[1] - y[0]) / dx
#     return dy

# # k = scipy.optimize.newton(lh, x0=1.7, fprime=dlh)
# k = scipy.optimize.newton(dlh, x0=1.7)
# theta = meand / k

# # pdf = rvs.Gamma(k, theta)

# x = np.linspace(0.01, 3, 1000)
# y = pdf.PDF(x)

# #p = np.linspace(0, 1, 8)[1:-1]
# #ppf = (-np.log(1-p) / lam)# + mthresh
# plt.clf()
# plt.hist(d, bins=100, normed=True, align='mid', range=[0,3])
# #plt.plot(x, y)
# #plt.plot(np.array([ppf, ppf]), np.array([[0]*ppf.size, [3.5]*ppf.size]), 'r--', linewidth=5)





# pdf1 = rvs.Gamma(4, 0.14)
# pdf2 = rvs.Gaussian(np.exp(0.7), np.exp(0.1))
# pdf3 = rvs.Gamma(1, 0.015)

# N = d.size * 3
# x1 = pdf1.sample(int(N * (3. / 6)))
# x2 = np.log(pdf2.sample(int(N * (2. / 6))))
# x3 = pdf3.sample(int(N * (1. / 6)))

# xx = list(x1) + list(x2) + list(x3)

# plt.clf()
# plt.subplot(1, 2, 1)
# plt.hist(xx, bins=100, normed=True, align='mid', range=[0,3])
# plt.ylim(0, 4.5)
# plt.subplot(1, 2, 2)
# plt.hist(d, bins=100, normed=True, align='mid', range=[0,3])
# plt.ylim(0, 4.5)
