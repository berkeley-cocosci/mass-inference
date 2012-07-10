import numpy as np

def resvec(vals, axis=None, ddof=0):
    alpha = np.array(vals, dtype='f8')
    # sum of cos & sin angles
    t = np.exp(1j * alpha)
    r = np.sum(t, axis=axis)
    # obtain length 
    r = np.abs(r) / alpha.shape[axis]
    # for data with known spacing, apply correction factor to correct for
    # bias in the estimation of r (see Zar, p. 601, equ. 26.16)
    if ddof != 0:
        r *= ddof / 2 / np.sin(ddof / 2)
    return r    

def nanresvec(vals, axis=None, ddof=0):
    alpha = np.array(vals, dtype='f8')
    # sum of cos & sin angles
    t = np.exp(1j * alpha)
    r = np.nansum(t, axis=axis)
    # obtain length 
    r = np.abs(r) / np.sum(~np.isnan(t), axis=axis)
    # for data with known spacing, apply correction factor to correct for
    # bias in the estimation of r (see Zar, p. 601, equ. 26.16)
    if ddof != 0:
        r *= ddof / 2 / np.sin(ddof / 2)
    return r


def mean(vals, axis=None):
    alpha = np.array(vals, dtype='f8')
    # sum of cos & sin angles
    t = np.exp(1j * alpha)
    r = np.sum(t, axis=axis)
    # obtain mean
    mu = np.angle(r) % (2.*np.pi)                        
    return mu

def nanmean(vals, axis=None):
    alpha = np.array(vals, dtype='f8')
    # sum of cos & sin angles
    t = np.exp(1j * alpha)
    r = np.nansum(t, axis=axis)
    # obtain mean
    mu = np.angle(r) % (2.*np.pi)
    return mu

def var(vals, axis=None, ddof=0):
    alpha = np.array(vals, dtype='f8')
    var = 1. - resvec(alpha, axis=axis, ddof=ddof)
    return var

def nanvar(vals, axis=None, ddof=0):
    alpha = np.array(vals, dtype='f8')
    var = 1. - nanresvec(alpha, axis=axis, ddof=ddof)
    return var

def std(vals, axis=None, ddof=0):
    var = var(vals, axis=axis, ddof=ddof)
    std = np.sqrt(-2*np.log(1 - var))
    return std

def nanstd(vals, axis=None, ddof=0):
    var = nanvar(vals, axis=axis, ddof=ddof)
    std = np.sqrt(-2*np.log(1 - var))
    return std
