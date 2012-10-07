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

def save(path, fignum=None, close=True, width=None, height=None):
    """Save a figure from pyplot"""
    if fignum is None:
        fig = plt.gcf()
    else:
        fig = plt.figure(fignum)

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

    print "Saving figure to '" + os.path.join(directory, filename) + "'...", 
    plt.savefig(os.path.join(directory, filename))

    if close:
        plt.clf()
        plt.cla()
        plt.close()
        
    print "Done"

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
    r, c = 3, 5
    n = r*c
    i = 0
    exp = np.exp(np.log(0.5) / np.log(1./27))
    if fignum is not None:
        plt.figure(fignum)
        plt.clf()
        plt.suptitle(
            "Posterior belief about mass ratio over time, "
            "$P_t(\kappa|F^{(r)})$, for various feedback conditions "
            "(%d samples, %s)" % (
                nsamps, "smoothed" if smooth else "unsmoothed"),
            fontsize=20)
        plt.subplots_adjust(
            wspace=0.1,
            hspace=0.3,
            left=0.08,
            right=0.95,
            top=0.92,
            bottom=0.05)

        kidxs = [0] + list(np.linspace(1, n_kappas-2, 13).astype('i8')) + [n_kappas-1]
        for kidx in kidxs:
            kappa = kappas[kidx]
            subjname = "Feedback ratio $r=%.2f$" % ratios[kidx]
            lat.plot_theta(
                r, c, i+1,
                np.exp(model_theta[kidx]),
                subjname,
                exp=exp,
                cmap=cmap)
            plt.plot(np.arange(n_trial),
                     np.ones(n_trial)*kidx,
                     '--', color="#ffff00",
                     linewidth=3, label="true ratio")

            yticks = np.round(np.linspace(0, n_kappas-1, 5)).astype('i8')
            if (i%c) == 0:
                plt.yticks(yticks, ratios[yticks], fontsize=12)
                plt.ylabel("Mass ratio ($\kappa$)")
            else:
                plt.yticks(yticks, [])
                plt.ylabel("")
            
            xticks = np.linspace(0, n_trial, 4).astype('i8')
            if (n-i) <= c:
                plt.xticks(xticks, xticks)
                plt.xlabel("Trial number ($t$)")
            else:
                plt.xticks(xticks, [])

            i += 1

        plt.legend()
        cticks = np.linspace(0, 1, 5)
        logcticks = np.round(np.exp(np.log(cticks) / np.log(exp)), decimals=4)
        cb = plt.colorbar(ticks=cticks)
        cb.set_ticklabels(logcticks)

    return model_lh, model_joint, model_theta
    
def plot_baserates(fignum, nsamps):

    plt.figure(fignum)
    plt.clf()
    plt.subplots_adjust(
        hspace=0.5,
        top=0.87,
        bottom=0.1,
        left=0.1,
        right=0.9,
        wspace=0.2)
    plt.suptitle(
        "Effect of model/feedback baserates on "
        "marginal posterior over mass ratio (%d samples)" % (nsamps),
        fontsize=20)
    
    i=0
    r, c = 4, 5
    n = r*c
    for nthresh0 in np.arange(0, 9, 2):
        for nthresh in np.arange(0, 9, 2):

            # feedback, ipe_samps = make_data(nthresh0, nthresh, nsamps)
            # ipe_baserate = np.swapaxes(ipe_samps, 0, 1).reshape(
            #     (n_kappas, -1)).mean(axis=1)
            # fb_baserate = np.swapaxes(feedback, 0, 1).reshape(
            #     (n_kappas, -1)).mean(axis=1)
            
            plt.subplot(r, c, i+1)
            # plt.plot(ipe_baserate,
            #          'r--',
            #          label="model baserate $P(fall|\kappa)$",
            #          linewidth=3)
            # plt.plot(fb_baserate,
            #          'm--',
            #          label="feedback baserate $P(fall|\kappa)$",
            #          linewidth=3)

            lh, jnt, th = run_and_plot(
                smooth=False,
                decay=1.0,
                fignum=None,
                nthresh0=nthresh0,
                nthresh=nthresh,
                nsamps=nsamps
                )

            exp = np.exp(np.log(0.5) / np.log(1. / n_kappas))
            plt.imshow(exp**normalize(th[:, -1].T, axis=-1)[1],
                       cmap=cmap,
                       interpolation='nearest')

            yticks = np.round(np.linspace(0, n_kappas-1, 5)).astype('i8')
            if (i%c) == 0:
                plt.yticks(yticks, ratios[yticks], fontsize=12)
                plt.ylabel("Mass ratio ($\kappa$)")
            else:
                plt.yticks(yticks, [])
                plt.ylabel("")
            
            xticks = np.round(np.linspace(0, n_kappas-1, 5)).astype('i8')
            if (n-i) <= c:
                plt.xticks(xticks, ratios[xticks], fontsize=12)
                plt.xlabel("Feedback condition ($r$)")
            else:
                plt.xticks(xticks, [])
                plt.xlabel("")

            # plt.plot(np.mean(np.exp(th[:, -1]), axis=0),
            #          'b-',
            #          label=r"unsmoothed marginal posterior $P_{t=384}(\kappa)$",
            #          linewidth=3)

            # lh, jnt, th = run_and_plot(
            #     smooth=True,
            #     decay=1.0,
            #     fignum=None,
            #     nthresh0=nthresh0,
            #     nthresh=nthresh,
            #     nsamps=nsamps
            #     )

            # plt.plot(np.mean(np.exp(th[:, -1]), axis=0),
            #          'c-',
            #          label=r"smoothed marginal posterior $P_{t=384}(\kappa)$",
            #          linewidth=3)


            # plt.xlim(0, n_kappas-1)
            # plt.ylim(0, 1)
            plt.title("Model \"fall\">%d blocks\n"
                      "Feedback \"fall\">%d blocks" % (
                          nthresh, nthresh0))

            # yticks = np.linspace(0, 1, 3)
            # if (i%c) == 0:
            #     plt.yticks(yticks, yticks)
            #     plt.ylabel("Probability")
            # else:
            #     plt.yticks(yticks, [])
            
            # xticks = (np.linspace(-1, 1, 3) + 1.3) * 10
            # xticklabels = np.round(10**np.linspace(-1, 1, 3), decimals=2)
            # if (n-i) <= c:
            #     plt.xticks(xticks, xticklabels)
            #     plt.xlabel("Mass ratio ($\kappa$)")
            # else:
            #     plt.xticks(xticks, [])

            #plt.grid(True)
            i+=1

    plt.legend(prop={'size': 12})

def plot_smoothing(fignum, nthresh, nsamps):

    samps = np.concatenate([
        ((rawipe['nfellA'] + rawipe['nfellB']) > nthresh).astype('int')[..., None],
        ], axis=-1)[..., 0].reshape(
        (rawipe.shape[0]/2, 2) + rawipe.shape[1:])[..., :nsamps]
    stims = np.array([int(x.split("_")[1])
                      for x in rawhstim]).reshape(
        (samps.shape[:2]))[:, 0]

    alpha = np.sum(samps, axis=-1) + 0.5
    beta = np.sum(1-samps, axis=-1) + 0.5
    pfell_mean = alpha / (alpha + beta)
    pfell_var = (alpha*beta) / ((alpha+beta)**2 * (alpha+beta+1))
    pfell_std = np.sqrt(pfell_var)
    pfell_meanstd = np.mean(pfell_std, axis=-1)

    colors = ['r', 'b']

    plt.figure(fignum)
    plt.clf()
    plt.suptitle(
        "Estimated likelihood of feedback given mass ratio "
        "for several pairs of stimuli (%d samples)" % (nsamps),
        fontsize=20)
    plt.subplots_adjust(
        hspace=0.2,
        wspace=0.2,
        top=0.9,
        bottom=0.1,
        left=0.07,
        right=0.95)


    r, c = 3, 4
    n = r*c
    for i in xrange(n):#n_trial):

        plt.subplot(r, c, i+1)
        plt.cla()
        
        for j in xrange(2):

            alpha = pfell_meanstd[i,j] * 10
            ell = 1. - np.std(pfell_mean[i,j])
            eps = pfell_meanstd[i,j] ** 2

            x = kappas
            xn = np.linspace(-1.5, 1.5, 100)
    
            gp = mo.make_gp(x, pfell_mean[i,j], alpha, ell, eps)
            y_mean, y_cov = gp(xn)

            plt.plot(xn, y_mean,
                     color=colors[j],
                     linewidth=3,
                     label="smoothing, assign %d" % (j+1))
            plt.errorbar(x, pfell_mean[i,j], pfell_std[i,j], None,
                         color=colors[j], fmt='o',
                         markeredgecolor=colors[j],
                         markersize=5,
                         label=("proportion \"fall\", "
                                "assign %d" % (j+1)))
            # plt.plot(x, pfell_mean[i,j], color=colors[j],
            #          linestyle='', marker='o',
            #          markeredgecolor=colors[j],
            #          markersize=5)

        plt.ylim(0, 1)
        plt.title("$S=$Tower %d" % (stims[i]))

        yticks = np.linspace(0, 1, 3)
        if (i%c) == 0:
            plt.yticks(yticks, yticks)
            plt.ylabel("$\Pr(fall|\kappa, S)$")
        else:
            plt.yticks(yticks, [])

        xticks = np.linspace(-1, 1, 3)
        if (n-i) <= c:
            plt.xticks(xticks, np.round(10**xticks, decimals=2))
            plt.xlabel("Mass ratio ($\kappa$)")
        else:
            plt.xticks(xticks, [])

        plt.grid(True)

    plt.legend(loc=0, prop={'size':12})

plot_baserates(1, nsamps=48)
save("images/baserates_048samples.png", width=18, height=10)
plot_baserates(3, nsamps=300)
save("images/baserates_300samples.png", width=18, height=10)

lh, jnt, th = run_and_plot(
    smooth=False,
    decay=1.0,
    fignum=5,
    nthresh0=0,
    nthresh=4,
    nsamps=48
    )
save("images/belief_raw_048samples.png", width=18, height=12)
lh, jnt, th = run_and_plot(
    smooth=True,
    decay=1.0,
    fignum=6,
    nthresh0=0,
    nthresh=4,
    nsamps=48
    )
save("images/belief_smoothed_048samples.png", width=18, height=12)
lh, jnt, th = run_and_plot(
    smooth=False,
    decay=1.0,
    fignum=7,
    nthresh0=0,
    nthresh=4,
    nsamps=300
    )
save("images/belief_raw_300samples.png", width=18, height=12)
lh, jnt, th = run_and_plot(
    smooth=True,
    decay=1.0,
    fignum=8,
    nthresh0=0,
    nthresh=4,
    nsamps=300
    )
save("images/belief_smoothed_300samples.png", width=18, height=12)


plot_smoothing(fignum=9, nthresh=4, nsamps=48)
save("images/likelihood_smoothing_048samples.png", width=18, height=10)
plot_smoothing(fignum=9, nthresh=4, nsamps=300)
save("images/likelihood_smoothing_300samples.png", width=18, height=10)




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
