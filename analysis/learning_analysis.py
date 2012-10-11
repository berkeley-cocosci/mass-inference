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

def save(path, fignum=None, close=True, width=None, height=None, ext=None):
    """Save a figure from pyplot"""
    if fignum is None:
        fig = plt.gcf()
    else:
        fig = plt.figure(fignum)

    if ext is None:
        ext = ['']

    if width:
        fig.set_figwidth(width)
    if height:
        fig.set_figheight(height)

    directory = os.path.split(path)[0]
    filename = os.path.split(path)[1]
    if directory == '':
        directory = '.'

    if not os.path.exists(directory):
        os.makedirs(directory)

    for ex in ext:
        if ex == '':
            name = filename
        else:
            name = filename + "." + ex

        print "Saving figure to %s...'" % (
            os.path.join(directory, name)),
        plt.savefig(os.path.join(directory, name))
        print "Done"

    if close:
        plt.clf()
        plt.cla()
        plt.close()

######################################################################
## Load and process data

rawhuman, rawhstim, raworder, rawtruth, rawipe, kappas = lat.load('stability')
#ratios = np.round(10 ** kappas, decimals=2)
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

def make_data(nthresh0, nthresh, nsamps):
    ipe_samps = np.concatenate([
        ((ipe['nfellA'] + ipe['nfellB']) > nthresh).astype('int')[..., None],
        ], axis=-1)[..., :nsamps, :]
    feedback = np.concatenate([
        ((truth['nfellA'] + truth['nfellB']) > nthresh0).astype('int'),
        ], axis=-1)
    return feedback, ipe_samps

######################################################################
# Model observers for each true mass ratio

cmap = make_cmap("lh", (0, 0, 0), (.5, .5, .5), (1, 0, 0))
vals = {}

def run_and_plot(smooth, decay, fignum, nthresh0, nthresh, nsamps):
    # memoize
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

    model_lh, model_joint, model_theta = vals[dparams][params]

    # plot it
    r, c = 3, 3
    n = r*c
    exp = np.exp(np.log(0.5) / np.log(1./27))    
    
    if fignum is not None:

        fig = plt.figure(fignum)
        plt.clf()
        gs = gridspec.GridSpec(r, c+1, width_ratios=[1]*c + [0.1])
        plt.suptitle(
            "Posterior belief about mass ratio over time\n"
            "(%d samples, %s, model thresh=%d blocks, fb thresh=%d blocks)" % (
                nsamps, "smoothed" if smooth else "unsmoothed", nthresh, nthresh0),
            fontsize=16)
        plt.subplots_adjust(
            wspace=0.2,
            hspace=0.3,
            left=0.1,
            right=0.93,
            top=0.85,
            bottom=0.1)

        # kidxs = ([0] +
        #          list(np.linspace(1, n_kappas-2, 13).astype('i8')) +
        #          [n_kappas-1])
        kidxs = [0, 3, 6, 10, 13, 16, 20, 23, 26]
        for i, kidx in enumerate(kidxs):
            irow, icol = np.unravel_index(i, (r, c))
            ax = plt.subplot(gs[irow, icol])
            kappa = kappas[kidx]
            subjname = "True $\kappa=%s$" % float(ratios[kidx])
            img = lat.plot_theta(
                #r, c, i+1,
                None, None, ax,
                np.exp(model_theta[kidx]),
                subjname,
                exp=exp,
                cmap=cmap,
                fontsize=14)
            # plt.plot(np.arange(n_trial),
            #          np.ones(n_trial)*kidx,
            #          '--', color="#ffff00",
            #          linewidth=3, label="true ratio")

            yticks = np.round(
                np.linspace(0, n_kappas-1, 5)).astype('i8')
            if (i%c) == 0:
                plt.yticks(yticks, ratios[yticks], fontsize=14)
                plt.ylabel("Mass ratio ($\kappa$)", fontsize=14)
            else:
                plt.yticks(yticks, [])
                plt.ylabel("")
            
            xticks = np.linspace(0, n_trial, 4).astype('i8')
            if (n-i) <= c:
                plt.xticks(xticks, xticks, fontsize=14)
                plt.xlabel("Trial number ($t$)", fontsize=14)
            else:
                plt.xticks(xticks, [])

        # plt.legend()
        # cticks = np.linspace(0, 1, 5)
        # logcticks = np.round(
        #     np.exp(np.log(cticks) / np.log(exp)), decimals=4)
        logcticks = np.array([0, 0.001, 0.05, 0.25, 1])
        cticks = np.exp(np.log(logcticks) * np.log(exp))
        cax = plt.subplot(gs[:, -1])
        cb = fig.colorbar(img, ax=ax, cax=cax, ticks=cticks)
        cb.set_ticklabels(logcticks)
        cax.set_title("$P_t(\kappa)$", fontsize=14)

    return model_lh, model_joint, model_theta
    
def plot_baserates(fignum, nsamps, smooth):

    plt.figure(fignum)
    plt.clf()
    plt.subplots_adjust(
        hspace=0.5,
        top=0.85,
        bottom=0.1,
        left=0.1,
        right=0.93,
        wspace=0.1)
    plt.suptitle(
        "Effect of model/feedback baserates on "
        "posterior over mass ratio\n(%d samples)" % (nsamps),
        fontsize=20)

    rows = np.array([0, 2, 4, 5])
    cols = np.array([0, 2, 4, 5])
    i=0
    r, c = rows.size, cols.size
    n = r*c

    for nthresh0 in rows:
        for nthresh in cols:
            
            lh, jnt, th = run_and_plot(
                smooth=True,
                decay=1.0,
                fignum=None,
                nthresh0=nthresh0,
                nthresh=nthresh,
                nsamps=nsamps
                )

            plt.subplot(r, c, i+1)
            exp = np.exp(np.log(0.5) / np.log(1. / n_kappas))
            nth = normalize(th[:, -1].T, axis=0)[1]
            post = np.mean(np.exp(nth), axis=-1)
            plt.imshow(exp**nth,
                       cmap=cmap,
                       interpolation='nearest')
            # plt.plot(post*(th.shape[0]-1),
            #          np.arange(post.size),
            #          'b', linewidth=2)
            plt.xlim(0, post.size-1)
            plt.ylim(0, post.size-1)

            yticks = np.round(
                np.linspace(0, n_kappas-1, 5)).astype('i8')
            if (i%c) == 0:
                plt.yticks(yticks, ratios[yticks], fontsize=12)
                plt.ylabel("Mass ratio ($\kappa$)")
            else:
                plt.yticks(yticks, [])
                plt.ylabel("")
            
            xticks = np.round(
                np.linspace(0, n_kappas-1, 5)).astype('i8')
            if (n-i) <= c:
                plt.xticks(xticks, ratios[xticks], fontsize=12)
                plt.xlabel("Feedback condition ($r$)")
            else:
                plt.xticks(xticks, [])
                plt.xlabel("")
            plt.title("Model thresh=%d\n"
                      "  Fb    thresh=%d" % (
                          nthresh, nthresh0),
                      fontsize=12)

            i+=1

def plot_smoothing(nstim, fignum, nthresh, nsamps):

    samps = np.concatenate([
        ((rawipe['nfellA'] + rawipe['nfellB']) > nthresh).astype(
            'int')[..., None]], axis=-1)[..., 0]
    stims = np.array([int(x.split("_")[1])
                      for x in rawhstim])

    alpha = np.sum(samps, axis=-1) + 0.5
    beta = np.sum(1-samps, axis=-1) + 0.5
    pfell_mean = alpha / (alpha + beta)
    pfell_var = (alpha*beta) / ((alpha+beta)**2 * (alpha+beta+1))
    pfell_std = np.sqrt(pfell_var)
    pfell_meanstd = np.mean(pfell_std, axis=-1)

    colors = cm.hsv(np.round(np.linspace(0, 220, nstim)).astype('i8'))
    xticks = np.linspace(-1.3, 1.3, 7)
    xticks10 = 10 ** xticks
    xticks10[xticks < 0] = np.round(xticks10[xticks < 0], decimals=2)
    xticks10[xticks >= 0] = np.round(xticks10[xticks >= 0], decimals=1)
    yticks = np.linspace(0, 1, 3)

    plt.figure(fignum)
    plt.clf()
    plt.suptitle(
        "Likelihood function for feedback given mass ratio\n"
        "(%d IPE samples, threshold=%d blocks)" % (nsamps, nthresh),
        fontsize=16)
    plt.ylim(-0.2, 1)
    plt.xticks(xticks, xticks10)
    plt.xlabel("Mass ratio ($\kappa$)", fontsize=14)
    plt.yticks(yticks, yticks)
    plt.ylabel("P(fall|$\kappa$, $S$)", fontsize=14)
    plt.grid(True)

    order = (range(0, stims.size, 2) + range(1, stims.size, 2))[:nstim]

    for idx in xrange(nstim):

        i = order[idx]
        x = kappas
        xn = np.linspace(-1.5, 1.5, 100)
    
        # alpha = pfell_meanstd[i] * 10
        # ell = 1. - np.std(pfell_mean[i,j])
        # eps = pfell_meanstd[i,j] ** 2
        # gp = mo.make_gp(x, pfell_mean[i,j], alpha, ell, eps)
        # y_mean, y_cov = gp(xn)

        lam = pfell_meanstd[i] * 10
        kde_smoother = mo.make_kde_smoother(x, pfell_mean[i,j], lam)
        y_mean = kde_smoother(xn)

        plt.plot(xn, y_mean,
                 color=colors[idx],
                 linewidth=3)        
        plt.errorbar(x, pfell_mean[i], pfell_std[i], None,
                     color=colors[idx], fmt='o',
                     markeredgecolor=colors[idx],
                     markersize=5,
                     label="Tower %d" % stims[i])

    plt.legend(loc=8, prop={'size':12}, numpoints=1,
               scatterpoints=1, ncol=3, title="Stimuli ($S$)")

ext = ['png', 'pdf']

plot_baserates(1, nsamps=48, smooth=True)
save("images/baserates_048samples",
     ext=ext, width=10, height=10)
plot_baserates(3, nsamps=300, smooth=True)
save("images/baserates_300samples",
     ext=ext, width=10, height=10)

nthresh0 = 1
nthresh = 4

lh, jnt, th = run_and_plot(
    smooth=False,
    decay=1.0,
    fignum=5,
    nthresh0=nthresh0,
    nthresh=nthresh,
    nsamps=48
    )
save("images/belief_raw_048samples",
     ext=ext, width=9, height=7)
lh, jnt, th = run_and_plot(
    smooth=True,
    decay=1.0,
    fignum=6,
    nthresh0=nthresh0,
    nthresh=nthresh,
    nsamps=48
    )
save("images/belief_smoothed_048samples",
     ext=ext, width=9, height=7)
lh, jnt, th = run_and_plot(
    smooth=False,
    decay=1.0,
    fignum=7,
    nthresh0=nthresh0,
    nthresh=nthresh,
    nsamps=300
    )
save("images/belief_raw_300samples",
     ext=ext, width=9, height=7)
lh, jnt, th = run_and_plot(
    smooth=True,
    decay=1.0,
    fignum=8,
    nthresh0=nthresh0,
    nthresh=nthresh,
    nsamps=300
    )
save("images/belief_smoothed_300samples",
     ext=ext, width=9, height=7)


plot_smoothing(6, fignum=9, nthresh=nthresh, nsamps=48)
save("images/likelihood_smoothing_048samples",
     ext=ext, width=9, height=7)
plot_smoothing(6, fignum=9, nthresh=nthresh, nsamps=300)
save("images/likelihood_smoothing_300samples",
     ext=ext, width=9, height=7)




# plt.clf()
# plt.fill_between(xn, above, below, color='r', alpha=0.2)
# plt.plot(xn, y_mean, color='k')
# plt.errorbar(kappas, pfell_mean[0], pfell_std[0], None, 'rx')
# plt.ylim(0, 1)



    
lh, jnt, th = run_and_plot(
    decay=1,   # weight decay rate
    h=2,       # gaussian kernel smoothing parameter
    u=0.0,       # mixture parameter for uniform component
    fignum=10,
    nthresh0=0,
    nthresh=4,
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
