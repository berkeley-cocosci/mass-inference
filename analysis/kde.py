import numpy as np
from scipy.integrate import trapz
from scipy.stats import nanmean
import pdb

import cogphysics.lib.rvs as rvs
import cogphysics.lib.nplib as npl
import cogphysics.lib.circ as circ

normalize = rvs.util.normalize
weightedSample = rvs.util.weightedSample

def to_lognorm(f):
    def transform(X, *args, **kwargs):
        hinv = np.log(X / (1.0 - X))
        d_hinv_dx = np.abs(1.0 / np.sum((X * (1.0 - X)), axis=-1))
        p_X = d_hinv_dx * f(hinv, *args, **kwargs)
        return p_X
    return transform

def to_rescaled(f, s, t):
    def transform(X, *args, **kwargs):
        hinv = (X - s) / float(t - s)
        d_hinv_dx = np.abs(1.0 / (t - s))
        p_X = d_hinv_dx * f(hinv, *args, **kwargs)
        return p_X
    return transform

def mv_gaussian_kde(vals, h=0.5):
    k = rvs.MVGaussian([0,0], np.eye(2))
    n = np.sum(np.sum(~np.isnan(vals), axis=-1), axis=-1)[..., None]
    def f(x):
        diff = x[..., None, :] - vals[..., None, :, :]
        pdf = np.ma.masked_invalid(k.PDF(diff / h))
        summed = pdf.filled(0.000001).sum(axis=-1)
        normed = summed / (n*h)
        return normed
    return f

def gaussian_kde(vals, h=0.5):
    k = rvs.Gaussian(0, 1)
    n = np.sum(~np.isnan(vals), axis=-1)[..., None]
    def f(x):
        diff = np.ma.masked_invalid(
            x[..., None] - vals[..., None, :])
        pdf = np.ma.masked_invalid(k.PDF(diff / h))
        summed = pdf.filled(0.000001).sum(axis=-1)
        normed = summed / (n*h)
        return normed
    return f

def vonmises_kde(vals, h=0.5):
    k = rvs.VonMises(0, 2*np.pi)
    #n = vals.shape[-1]
    n = np.sum(~np.isnan(vals), axis=-1)[..., None]
    def f(x):
        diff = np.ma.masked_invalid(
            circ.difference(x[..., None], vals[..., None, :]))
        #pdf = k.PDF(diff.filled(0) / h)
        pdf = np.ma.masked_invalid(k.PDF(diff / h))
        summed = pdf.filled(0.01).sum(axis=-1)
        normed = summed / (n*h)
        return normed
    return f

def logit(x, s, t):
    r = (x - s).astype('f8') / (t - s)
    l = np.log(r / (1.0 - r))
    return l

def makeBins(vmin, vmax, nbins):
    edges = np.linspace(vmin, vmax, nbins+1)
    mids = (edges[:-1] + edges[1:]) / 2.
    binsize = float(vmax - vmin) / nbins
    return edges, mids, binsize

def gen_stability_edges(n):
    edges, mids, binsize = makeBins(-0.5, 10.5, n)
    return edges, binsize

def gen_radius_edges(n):
    edges, mids, binsize = makeBins(-.015, .315, n)
    return edges, binsize

def gen_xy_edges(n, which='xy'):
    if which == 'xy':
        xedges, xmids, xbinsize = makeBins(-.315, .315, n[0])
        yedges, ymids, ybinsize = makeBins(-.315, .315, n[1])
        edges = (xedges, yedges)
        binsize = (xbinsize, ybinsize)
    elif which == 'x':
        xedges, xmids, xbinsize = makeBins(-.315, .315, n)
        edges = xedges
        binsize = xbinsize
    elif which == 'y':
        yedges, ymids, ybinsize = makeBins(-.315, .315, n)
        edges = yedges
        binsize = ybinsize
    return edges, binsize

def gen_direction_edges(n):
    offset = np.pi / n
    edges, mids, binsize = makeBins(offset, 2*np.pi + offset, n)
    return edges, binsize, offset

def stability_nfell_kde(vals, n, h=0.2, s=-5, t=15):
    lvals = logit(vals, s, t)
    f = gaussian_kde(lvals, h=h)
    x, binsize = gen_stability_edges(n)
    px = to_rescaled(to_lognorm(f), s, t)(x)
    bx = normalize(np.log(trapz(
        np.array([px[..., :-1], px[..., 1:]]),
        dx=binsize, axis=0)), axis=-1)[1]
    return bx

def direction_kde(vals, n, h=1.0):
    f = vonmises_kde(vals.copy(), h=h)
    x, binsize, offset = gen_direction_edges(n)
    px = f(x)
    bx = normalize(np.log(trapz(
        np.array([px[..., :-1], px[..., 1:]]),
        dx=binsize, axis=0)), axis=-1)[1]
    return bx

def radius_kde(vals, n, h=0.2, s=-.15, t=.45):
    lvals = logit(vals, s, t)
    f = gaussian_kde(lvals, h=h)
    x, binsize = gen_radius_edges(n)
    px = to_rescaled(to_lognorm(f), s, t)(x)
    bx = normalize(np.log(trapz(
        np.array([px[..., :-1], px[..., 1:]]),
        dx=binsize, axis=0)), axis=-1)[1]
    return bx

def xy_kde(x_vals, y_vals, n, h=0.2, s=-.35, t=.35):

    vals = np.concatenate([x_vals[..., None], y_vals[..., None]], axis=-1)
    lvals = logit(vals, s, t)
    f = mv_gaussian_kde(lvals, h=h)
    (x0, x1), (b0, b1) = gen_xy_edges(n)
    x = np.array(np.meshgrid(x0, x1)).T.reshape((-1, 2))
    sqshape = vals.shape[:-2] + x0.shape + x1.shape
    px = to_rescaled(to_lognorm(f), s, t)(x).reshape(sqshape)
    i0 = trapz(np.array([px[..., :-1, :], px[..., 1:, :]]),
               dx=b0, axis=0)
    i1 = trapz(np.array([i0[..., :-1], i0[..., 1:]]),
               dx=b1, axis=0)
    flatshape = i1.shape[:-2] + (np.prod(i1.shape[-2:]),)
    sqshape = i1.shape
    bx = normalize(np.log(i1.reshape(flatshape)), axis=-1)[1].reshape(sqshape)

    #f0 = gaussian_kde(lvals[..., 0, :], h=h)
    #f1 = gaussian_kde(lvals[..., 1, :], h=h)
    #px0 = to_rescaled(to_lognorm(f0), s, t)(x0)
    #px1 = to_rescaled(to_lognorm(f1), s, t)(x1)
    # i0 = normalize(np.log(trapz(
    #     np.array([px0[..., :-1], px0[..., 1:]]),
    #     dx=b0, axis=0)), axis=-1)[1]
    # i1 = normalize(np.log(trapz(
    #     np.array([px1[..., :-1], px1[..., 1:]]),
    #     dx=b1, axis=0)), axis=-1)[1]
    # bx = i0[..., :, None] + i1[..., None, :]

    # import matplotlib.pyplot as plt
    # plt.plot(np.arange(i0.shape[2])-0.5, i0.ravel())
    # plt.plot(i1.ravel(), np.arange(i1.shape[2])-0.5)
    # pdb.set_trace()

    return bx
