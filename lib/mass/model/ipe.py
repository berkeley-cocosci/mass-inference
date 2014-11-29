import numpy as np
import pandas as pd
import snippets.datapackage as dpkg
import snippets.circstats as cs
import scipy.special
import scipy.stats
import seaborn as sns

from mass import DATA_PATH
from .util import LazyProperty
from ..analysis import bootstrap_mean


class IPE(object):
    def __init__(self, name, sigma, phi):
        self.path = DATA_PATH.joinpath("model/%s.dpkg" % name)
        self.dp = dpkg.DataPackage.load(self.path)
        self.data = self.dp\
                        .load_resource("model.csv")\
                        .set_index(['sigma', 'phi', 'stimulus'])
        self.sigma = float(sigma)
        self.phi = float(phi)

    def _sample_kappa_mean(self, data):
        samps = data.pivot(
            index='sample',
            columns='kappa',
            values='nfell')
        mean = (samps / 10.0).mean()
        return mean

    def _sample_kappa_var(self, data):
        samps = data.pivot(
            index='sample',
            columns='kappa',
            values='nfell')
        var = (samps / 10.0).var()
        return var

    def _sample_kappa_sem(self, data):
        samps = data.pivot(
            index='sample',
            columns='kappa',
            values='nfell')
        var = (samps / 10.0).apply(scipy.stats.sem)
        return var

    def _sample_kappa_stats(self, data):
        samps = data.pivot(
            index='sample',
            columns='kappa',
            values='nfell')
        sem = (samps / 10.0).apply(bootstrap_mean)
        return sem

    def _bincount(self, x):
        counts = np.bincount(np.asarray(x, dtype=int))
        all_counts = np.zeros(11)
        all_counts[:len(counts)] = counts
        return pd.Series(all_counts, index=np.arange(0, 11), name=x.name)

    def _sample_kappa_nfell(self, data):
        samps = data.pivot(
            index='sample',
            columns='kappa',
            values='nfell')
        counts = samps.apply(self._bincount, axis=0) + 1
        counts = counts / counts.sum(axis=0)
        counts.index.name = 'nfell'
        return counts

    def _sample_kappa_dir_mean(self, data):
        samps = data.pivot(
            index='sample',
            columns='kappa',
            values='direction')
        samps_arr = np.asarray(samps)
        # Note, nanmean prints the following warning: "FutureWarning:
        # In Numpy 1.9 the sum along empty slices will be zero." This
        # happens when there is a column in samps_arr that is entirely
        # NaNs. So we check for that and explicitly set it to NaN, for
        # future compatibility.
        mean = pd.Series(
            cs.nanmean(samps_arr, axis=0),
            index=samps.columns)
        isnan = np.isnan(samps_arr).all(axis=0)
        mean[isnan] = np.nan
        mean.name = data.name
        return mean

    def _sample_kappa_dir_var(self, data):
        samps = data.pivot(
            index='sample',
            columns='kappa',
            values='direction')
        samps_arr = np.asarray(samps)
        # Note, nankappa prints the following warning: "FutureWarning:
        # In Numpy 1.9 the sum along empty slices will be zero." This
        # happens when there is a column in samps_arr that is entirely
        # NaNs. So we check for that and explicitly set it to NaN, for
        # future compatibility.
        var = pd.Series(
            cs.nankappa(samps_arr, axis=0),
            index=samps.columns)
        isnan = np.isnan(samps_arr).all(axis=0)
        var[isnan] = np.nan
        var.name = data.name
        return var

    @LazyProperty
    def P_fall_mean_all(self):
        return self.data\
                   .groupby(level=['sigma', 'phi', 'stimulus'])\
                   .apply(self._sample_kappa_mean)

    @LazyProperty
    def P_fall_mean(self):
        return self.P_fall_mean_all\
                   .groupby(level=['sigma', 'phi'])\
                   .get_group((self.sigma, self.phi))\
                   .reset_index(['sigma', 'phi'], drop=True)

    @LazyProperty
    def P_fall_sem_all(self):
        return self.data\
                   .groupby(level=['sigma', 'phi', 'stimulus'])\
                   .apply(self._sample_kappa_sem)

    @LazyProperty
    def P_fall_sem(self):
        return self.P_fall_sem_all\
                   .groupby(level=['sigma', 'phi'])\
                   .get_group((self.sigma, self.phi))\
                   .reset_index(['sigma', 'phi'], drop=True)

    @LazyProperty
    def P_fall_stats(self):
        stats = self.data\
                   .groupby(level=['sigma', 'phi'])\
                   .get_group((self.sigma, self.phi))\
                   .reset_index(['sigma', 'phi'], drop=True)\
                   .groupby(level='stimulus')\
                   .apply(self._sample_kappa_stats)
        stats.index.names = ['stimulus', 'stat']
        return stats

    @LazyProperty
    def P_fall_var_all(self):
        return self.data\
                   .groupby(level=['sigma', 'phi', 'stimulus'])\
                   .apply(self._sample_kappa_var)

    @LazyProperty
    def P_fall_var(self):
        return self.P_fall_var_all\
                   .groupby(level=['sigma', 'phi'])\
                   .get_group((self.sigma, self.phi))\
                   .reset_index(['sigma', 'phi'], drop=True)

    @LazyProperty
    def P_nfell_all(self):
        return self.data\
                   .groupby(level=['sigma', 'phi', 'stimulus'])\
                   .apply(self._sample_kappa_nfell)

    @LazyProperty
    def P_nfell(self):
        return self.P_nfell_all\
                   .groupby(level=['sigma', 'phi'])\
                   .get_group((self.sigma, self.phi))\
                   .reset_index(['sigma', 'phi'], drop=True)

    @LazyProperty
    def P_dir_mean_all(self):
        return self.data\
            .groupby(level=['sigma', 'phi', 'stimulus'])\
            .apply(self._sample_kappa_dir_mean)

    @LazyProperty
    def P_dir_mean(self):
        return self.P_dir_mean_all\
            .groupby(level=['sigma', 'phi'])\
            .get_group((self.sigma, self.phi))\
            .reset_index(['sigma', 'phi'], drop=True)

    @LazyProperty
    def P_dir_var_all(self):
        return self.data\
            .groupby(level=['sigma', 'phi', 'stimulus'])\
            .apply(self._sample_kappa_dir_var)

    @LazyProperty
    def P_dir_var(self):
        return self.P_dir_var_all\
            .groupby(level=['sigma', 'phi'])\
            .get_group((self.sigma, self.phi))\
            .reset_index(['sigma', 'phi'], drop=True)

    @LazyProperty
    def P_fall_smooth(self):
        mean = self.P_fall_mean

        # sanity checks
        assert mean.index.name == 'stimulus'
        assert mean.columns.name == 'kappa'

        # TODO: this value is kind of arbitrary
        lam = 0.2

        # shape is (n_kappas, n_kappas)
        kappas = np.array(map(float, mean.columns))
        dists = np.abs(kappas[:, None] - kappas[None, :]) / lam
        # calculate gaussian probability of distances
        pdist = np.exp(-0.5 * dists ** 2) / np.sqrt(2.0)
        # shape is (n_kappas,)
        sum_pdist = np.sum(pdist, axis=-1)
        # broadcasts to shape (n_trial, n_kappas, n_kappas), then sum
        # leads to shape of (n_trial, n_kappas)
        pfell_arr = np.asarray(mean)[..., None, :]
        smooth_arr = np.sum(pdist * pfell_arr, axis=-1) / sum_pdist

        # convert back to DataFrame
        smooth = pd.DataFrame(
            smooth_arr,
            index=mean.index,
            columns=mean.columns)

        return smooth

    def plot_fall(self, ax, ix, color='b'):
        if isinstance(ix, int):
            ix = self.P_fall_smooth.index[ix]

        smooth = self.P_fall_smooth.ix[ix]
        mean = self.P_fall_mean.ix[ix]
        var = self.P_fall_var.ix[ix]
        std = var.map(np.sqrt)

        darkgrey = "#404040"

        ax.plot(
            smooth.index, smooth,
            lw=2, color=darkgrey)
        ax.errorbar(
            mean.index, mean, yerr=std,
            ls='', marker='o', color=color, ecolor=darkgrey)

        ax.set_xlabel("kappa")
        ax.set_ylabel("P(fall | kappa)")
        ax.set_title(r"$\sigma$=%.2f, $\phi$=%.2f" % (self.sigma, self.phi)
                     + "\nstim=%s" % ix)
        ax.set_ylim(0, 1)

    def plot_direction(self, ax, ix, kappa, r=1, color='b'):
        if isinstance(ix, int):
            ix = self.P_dir_mean.index[ix]

        mean = self.P_dir_mean.ix[ix][kappa]
        var = self.P_dir_var.ix[ix][kappa]

        points = self\
            .data\
            .reset_index()\
            .groupby(['sigma', 'phi', 'kappa'])\
            .get_group((self.sigma, self.phi, kappa))\
            .pivot('stimulus', 'sample', 'direction')\
            .ix[ix]\
            .dropna()

        X = np.linspace(0, 2 * np.pi, 1000)
        Y = cs.vmpdf(X, mean, var) + r

        ax.plot(X, Y, lw=2, color=color)
        ax.plot(X, np.ones(X.size) * r, '-', color='#404040')
        ax.plot(points, np.ones(points.size) * r, 'o', color=color)

        ax.set_title(
            r"$\sigma$=%.2f, $\phi$=%.2f, $\kappa$=%.2f" % (
                self.sigma, self.phi, kappa)
            + "\nstim=%s" % ix)

        ax.spines['polar'].set_color('none')
        ax.grid('off', axis='y')
