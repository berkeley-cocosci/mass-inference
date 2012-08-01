import numpy as np
from scipy.integrate import trapz

import cogphysics.lib.rvs as rvs
import cogphysics.lib.nplib as npl
import cogphysics.lib.circ as circ

normalize = rvs.util.normalize
weightedSample = rvs.util.weightedSample

def to_lognorm(f):
    def transform(X, *args, **kwargs):
        hinv = np.log(X / (1.0 - X))
        d_hinv_dx = np.abs(1.0 / (X * (1.0 - X)))
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

def gaussian_kde(vals, h=0.5):
    k = rvs.Gaussian(0, 1)
    n = vals.shape[-1]
    def f(x):
        diff = x[..., None] - vals[..., None, :]
        pdf = k.PDF(diff / h)
        summed = np.sum(pdf, axis=-1)
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

def radius_kde(vals, n, h=0.2, s=-.15, .45):
    lvals = logit(vals, s, t)
    f = gaussian_kde(lvals, h=h)
    x, binsize = gen_radius_edges(n)
    px = to_rescaled(to_lognorm(f), s, t)(x)
    bx = normalize(np.log(trapz(
        np.array([px[..., :-1], px[..., 1:]]),
        dx=binsize, axis=0)), axis=-1)[1]
    return bx
