import numpy as np
import scipy.special
import pandas as pd

from .util import LazyProperty


class EmpiricalIPE(object):
    def __init__(self, data):
        self.data = data.copy().rename(columns={'kappa0': 'kappa'})
        self.data['fall? response'] = self.data['fall? response'] - 1

    def _sample_kappa_alpha(self, data):
        samps = data.pivot(
            index='pid',
            columns='kappa',
            values='fall? response')
        alpha = samps.sum() + 0.5
        return alpha

    def _sample_kappa_beta(self, data):
        samps = data.pivot(
            index='pid',
            columns='kappa',
            values='fall? response')
        beta = (6 - samps).sum() + 0.5
        return beta

    @LazyProperty
    def alpha(self):
        return self.data.groupby('stimulus').apply(self._sample_kappa_alpha)

    @LazyProperty
    def beta(self):
        return self.data.groupby('stimulus').apply(self._sample_kappa_beta)

    @LazyProperty
    def P_fall_mean(self):
        alpha = self.alpha
        beta = self.beta
        mean = alpha / (alpha + beta)
        return mean

    @LazyProperty
    def P_fall_var(self):
        alpha = self.alpha
        beta = self.beta
        var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        return var

    @LazyProperty
    def P_fall_stats(self):
        alpha = self.alpha.stack()
        beta = self.beta.stack()
        stats = scipy.special.btdtri(
            np.asarray(alpha)[:, None],
            np.asarray(beta)[:, None],
            [0.025, 0.5, 0.975])
        stats = pd.DataFrame(
            stats,
            index=alpha.index,
            columns=['lower', 'median', 'upper'])
        stats.columns.name = 'stat'
        stats = stats\
            .stack()\
            .unstack('kappa')
        return stats

    @property
    def P_fall_smooth(self):
        return self.P_fall_mean

    def plot_fall(self, ax, ix):
        if isinstance(ix, int):
            ix = self.P_fall_smooth.index[ix]

        smooth = self.P_fall_smooth.ix[ix]
        smooth_index = smooth.index
        smooth = np.asarray(smooth)

        mean = self.P_fall_mean.ix[ix]
        mean_index = mean.index
        mean = np.asarray(mean)

        var = np.asarray(self.P_fall_var.ix[ix])
        std = np.sqrt(var)

        ax.plot(
            smooth_index, smooth,
            lw=2, color='k')
        ax.errorbar(
            mean_index, mean, yerr=std,
            ls='', marker='o', color='b', ecolor='k')

        ax.set_xlabel("kappa")
        ax.set_ylabel("P(fall | kappa)")
        ax.set_title(r"stim=%s" % ix)
        ax.set_ylim(0, 1)
