import numpy as np
import circstats as circ
import scipy.optimize

from joblib import Memory
memory = Memory(cachedir="cache", mmap_mode='c', verbose=0)


def normalize(logarr, axis=-1, max_log_value=709.78271289338397):
    """Normalize an array of log-values.  Returns a tuple of
    (normalization constants, normalized array), where both values are
    again in logspace.

    """

    # shape for the normalization constants (that would otherwise be
    # missing axis)
    shape = list(logarr.shape)
    shape[axis] = 1
    # filter out infinite values
    x = np.ma.masked_invalid(logarr)
    # get maximum value of array
    maxlogarr = x.max(axis=axis).reshape(shape)
    # calculate how much to shift the array up by
    shift = (max_log_value - maxlogarr - 2 - logarr.shape[axis])
    # shift the array
    unnormed = logarr + shift
    # convert from logspace
    arr = np.ma.exp(unnormed)
    # calculate shifted log normalization constants
    _lognormconsts = np.ma.log(arr.sum(axis=axis)).reshape(shape)
    # calculate normalized array
    lognormarr = np.ma.filled(unnormed - _lognormconsts, fill_value=-np.inf)
    # unshift normalization constants
    _lognormconsts -= shift
    # get rid of the dimension we normalized over
    lognormconsts = np.ma.filled(
        _lognormconsts.sum(axis=axis),
        fill_value=-np.inf)

    return lognormconsts, lognormarr


def xcorr(x, y, circular=False, deg=False, nanrobust=False):
    """Returns matrix of correlations between x and y

    Parameters
    ----------
    x : np.ndarray
        Columns are different parameters, rows are sets of observations
    y : np.ndarray
        Columns are different parameters, rows are sets of observations
    circular : bool (default=False)
        Whether or not the data is circular
    deg : bool (default=False)
        Whether or not the data is in degrees (if circular)

    Returns
    -------
    out : np.ndarray
        Matrix of correlations between rows of x and y
    """

    # Inputs' shapes
    xshape = x.shape
    yshape = y.shape

    # Store original (output) shapes
    corrshape = xshape[:-1] + yshape[:-1]

    # Prepares inputs' shapes for computations
    if len(x.shape) > 2:
        x = x.reshape((np.prod(xshape[:-1]), xshape[-1]), order='C')
    if len(y.shape) > 2:
        y = y.reshape((np.prod(yshape[:-1]), yshape[-1]), order='C')

    if x.ndim == 1:
        x = x[None, :]
    if y.ndim == 1:
        y = y[None, :]

    if circular:
        if deg:
            x = np.radians(x)
            y = np.radians(y)

        if nanrobust:
            corr = circ.nancorrcc(
                x[:, :, None], y.T[None, :, :], axis=1)
        else:
            corr = circ.corrcc(
                x[:, :, None], y.T[None, :, :], axis=1)

    else:
        avgfn = np.mean
        stdfn = np.std

        # numerator factors (centered means)
        nx = (x.T - avgfn(x, axis=1)).T
        ny = (y.T - avgfn(y, axis=1)).T

        # denominator factors (std devs)
        sx = stdfn(x, axis=1)
        sy = stdfn(y, axis=1)

        # numerator
        num = np.dot(nx, ny.T) / x.shape[1]

        # correlation
        corr = num / np.outer(sx, sy)

    # reshape to take original
    corr = corr.reshape(corrshape, order='F')

    return corr


def logit(x, alpha, beta, omega):
    l = (alpha / (1 + np.e ** (-beta * x))) + omega
    return l


@memory.cache
def fit_logit(x, y, start):
    params = np.empty((y.shape[0], 3)) * np.nan
    for i in xrange(params.shape[0]):
        try:
            popt, pcov = scipy.optimize.curve_fit(
                logit, x, y[i], start)
        except:
            pass
        else:
            params[i] = popt
    return params
