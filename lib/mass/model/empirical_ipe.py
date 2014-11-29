import numpy as np
import scipy.special
import pandas as pd
import seaborn as sns

from .util import LazyProperty
from mass.analysis import bootstrap_mean


class EmpiricalIPE(object):
    def __init__(self, data):
        self.data = data.copy().rename(columns={'kappa0': 'kappa'})
        self.data['fall? response'] = (self.data['fall? response'] - 1) / 6.0

    def _sample_kappa_mean(self, data):
        samps = data.pivot(
            index='pid',
            columns='kappa',
            values='fall? response')
        return samps.mean()

    def _sample_kappa_var(self, data):
        samps = data.pivot(
            index='pid',
            columns='kappa',
            values='fall? response')
        return samps.var()

    def _sample_kappa_sem(self, data):
        samps = data.pivot(
            index='pid',
            columns='kappa',
            values='fall? response')
        return samps.apply(scipy.stats.sem)

    def _sample_kappa_stats(self, data):
        samps = data.pivot(
            index='pid',
            columns='kappa',
            values='fall? response')
        return samps.apply(bootstrap_mean)

    @LazyProperty
    def P_fall_mean(self):
        return self.data.groupby('stimulus').apply(self._sample_kappa_mean)

    @LazyProperty
    def P_fall_var(self):
        return self.data.groupby('stimulus').apply(self._sample_kappa_var)

    @LazyProperty
    def P_fall_sem(self):
        return self.data.groupby('stimulus').apply(self._sample_kappa_sem)

    @LazyProperty
    def P_fall_stats(self):
        stats = self.data.groupby('stimulus').apply(self._sample_kappa_stats)
        stats.index.names = ['stimulus', 'stat']
        return stats

    @property
    def P_fall_smooth(self):
        return self.P_fall_mean

    def plot_fall(self, ax, ix, color='b'):
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

        darkgrey = "#404040"

        ax.plot(
            smooth_index, smooth,
            lw=2, color=darkgrey)
        ax.errorbar(
            mean_index, mean, yerr=std,
            ls='', marker='o', color=color, ecolor=darkgrey)

        ax.set_xlabel("kappa")
        ax.set_ylabel("P(fall | kappa)")
        ax.set_title(r"stim=%s" % ix)
        ax.set_ylim(0, 1)
