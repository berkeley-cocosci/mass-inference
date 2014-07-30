import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import snippets.datapackage as dpkg
import snippets.circstats as cs

from mass import DATA_PATH
from .util import LazyProperty


class IPE(object):
    def __init__(self, name, sigma, phi):
        self.path = DATA_PATH.joinpath("model/%s.dpkg" % name)
        self.dp = dpkg.DataPackage.load(self.path)
        self.data = self.dp\
                        .load_resource("model.csv")\
                        .set_index(['sigma', 'phi', 'stimulus'])
        self.sigma = float(sigma)
        self.phi = float(phi)

    def _sample_kappa_fall_mean(self, data):
        samps = data.pivot(
            index='sample',
            columns='kappa',
            values='nfell')
        alpha = samps.sum() + 0.5
        beta = (10 - samps).sum() + 0.5
        mean = alpha / (alpha + beta)
        return mean

    def _sample_kappa_fall_var(self, data):
        samps = data.pivot(
            index='sample',
            columns='kappa',
            values='nfell')
        alpha = samps.sum() + 0.5
        beta = (10 - samps).sum() + 0.5
        var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        return var

    def _sample_kappa_dir_mean(self, data):
        samps = data.pivot(
            index='sample',
            columns='kappa',
            values='direction')
        mean = pd.Series(
            cs.nanmean(np.asarray(samps), axis=0),
            index=samps.columns)
        mean.name = data.name
        return mean

    def _sample_kappa_dir_var(self, data):
        samps = data.pivot(
            index='sample',
            columns='kappa',
            values='direction')
        var = pd.Series(
            cs.nankappa(np.asarray(samps), axis=0),
            index=samps.columns)
        var.name = data.name
        return var

    @LazyProperty
    def P_fall_mean_all(self):
        return self.data\
                   .groupby(level=['sigma', 'phi', 'stimulus'])\
                   .apply(self._sample_kappa_fall_mean)

    @LazyProperty
    def P_fall_mean(self):
        return self.P_fall_mean_all\
                   .groupby(level=['sigma', 'phi'])\
                   .get_group((self.sigma, self.phi))\
                   .reset_index(['sigma', 'phi'], drop=True)

    @LazyProperty
    def P_fall_var_all(self):
        return self.data\
                   .groupby(level=['sigma', 'phi', 'stimulus'])\
                   .apply(self._sample_kappa_fall_var)

    @LazyProperty
    def P_fall_var(self):
        return self.P_fall_var_all\
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

    def plot_fall(self, ix):
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
        ax.set_title(r"$\sigma$=%.2f, $\phi$=%.2f, stim=%s" % (
            self.sigma, self.phi, ix))
        ax.set_ylim(0, 1)

    def plot_direction(self, ix, kappa):
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

        r = 0.5
        X = np.linspace(0, 2 * np.pi, 1000)
        Y = cs.vmpdf(X, mean, var) + r

        ax = plt.subplot(111, polar=True)
        ax.plot(X, Y, lw=2, color='k')
        ax.plot(points, np.ones(points.size) * r, 'ro')

        ax.set_title(r"$\sigma$=%.2f, $\phi$=%.2f, $\kappa$=%.2f, stim=%s" % (
            self.sigma, self.phi, kappa, ix))
