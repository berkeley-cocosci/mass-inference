import matplotlib.pyplot as plt
import numpy as np

from .util import LazyProperty


class EmpiricalIPE(object):
    def __init__(self, data):
        self.data = data.copy().rename(columns={'kappa0': 'kappa'})
        self.data['fall? response'] = (self.data['fall? response'] - 1) / 6.

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
        var = samps.var()
        n = samps.count()
        return var / n

    @LazyProperty
    def P_fall_mean(self):
        return self.data.groupby('stimulus').apply(self._sample_kappa_mean)

    @LazyProperty
    def P_fall_var(self):
        return self.data.groupby('stimulus').apply(self._sample_kappa_var)

    @property
    def P_fall_smooth(self):
        return self.P_fall_mean

    def plot(self, ix):
        if isinstance(ix, int):
            ix = self.P_fall_smooth.index[ix]

        smooth = self.P_fall_smooth.ix[ix]
        mean = self.P_fall_mean.ix[ix]
        var = self.P_fall_var.ix[ix]
        std = var.map(np.sqrt)

        fig, ax = plt.subplots()
        ax.plot(
            smooth.index, smooth,
            lw=2, color='k')
        ax.errorbar(
            mean.index, mean, yerr=std,
            ls='', marker='o', color='b', ecolor='k')

        ax.set_xlabel("kappa")
        ax.set_ylabel("P(fall | kappa)")
        ax.set_title(r"stim=%s" % ix)
        ax.set_ylim(0, 1)
