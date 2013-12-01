import pymc
import numpy as np
import IPython
import matplotlib.pyplot as plt
import pandas as pd

from pymc.distributions import binomial_like
from .util import LazyProperty


class FallModel(object):

    # names of the keys that get stored when the model is pickled
    _state_keys = [
        'trials',
        'n_trial',
        'kappas',
        'n_kappa',
        'ipe_mean',
        'ipe_var',
        'ipe_smooth',
        'exp',
        'prior',
        'pymc_values'
    ]

    def __init__(self, ipe, exp, trials, kappas=None, prior=None):
        self.trials = map(str, trials)
        self.n_trial = len(self.trials)

        if kappas is None:
            self.kappas = map(float, ipe.P_fall_mean.columns)
        else:
            self.kappas = map(float, kappas)
        self.n_kappa = len(self.kappas)

        self.ipe_mean = ipe.P_fall_mean.ix[self.trials][self.kappas]
        self.ipe_var = ipe.P_fall_var.ix[self.trials][self.kappas]
        self.ipe_smooth = ipe.P_fall_smooth.ix[self.trials][self.kappas]
        self.exp = exp.ix[self.trials]

        if prior is None:
            self.prior = np.log(np.ones(self.n_kappa) / float(self.n_kappa))
        else:
            if len(prior) != self.n_kappa:
                raise ValueError("given prior is not the right shape")
            self.prior = np.log(prior)

        self.init_pymc_variables()

    def init_pymc_variables(self):
        self.variables = [self.S, self.k, self.p_fall, self.J]

    @property
    def pymc_values(self):
        # we don't need S or p_fall because they are deterministic, or
        # J because it is observed
        values = {
            'k': self.k.value
        }
        return values

    @pymc_values.setter
    def pymc_values(self, values):
        self.k.value = values['k']

    def __getstate__(self):
        state = {key: getattr(self, key) for key in self._state_keys}
        return state

    def __setstate__(self, state):
        for key in self._state_keys:
            # we have to set these values AFTER we've initialized the
            # pymc variables
            if key == 'pymc_values':
                continue
            setattr(self, key, state[key])

        self.init_pymc_variables()
        self.pymc_values = state['pymc_values']

    @LazyProperty
    def S(self):
        """Stimuli"""
        S = np.empty(self.n_trial, dtype=object)
        for t, stim in enumerate(self.trials):
            @pymc.deterministic(plot=False, trace=False, name="S_%d" % t)
            def S_t():
                return stim
            S[t] = S_t
        return S

    @LazyProperty
    def k(self):
        k = pymc.Categorical('k', np.exp(self.prior))
        return k

    @LazyProperty
    def p_fall(self):
        """Probability that the tower falls, independent of mass"""
        p_fall = np.empty(self.n_trial, dtype=object)
        k = self.k
        for t, S in enumerate(self.S):
            @pymc.deterministic(name="p_fall_%d" % t)
            def p_fall_t(k=k, S=S):
                kappa = self.kappas[int(k)]
                return self.ipe_smooth.ix[S, kappa]
            p_fall[t] = p_fall_t
        return p_fall

    @LazyProperty
    def J(self):
        """Bernoulli judgment of whether the tower will fall"""
        J = np.empty(self.n_trial, dtype=object)
        for t in xrange(self.n_trial):
            p_fall = self.p_fall[t]
            S = self.S[t].value
            value = self.exp[S] - 1
            J[t] = pymc.Binomial(
                'J_%d' % t, 6, p_fall,
                value=value, observed=True)
        return J

    @property
    def logp(self):
        m = pymc.Model(self)
        return m.logp

    @property
    def logp_k(self):
        logp = np.empty(self.n_kappa)
        for i in np.arange(self.n_kappa):
            self.k.value = i
            logp[i] = self.logp
        s = pd.Series(logp, index=self.kappas)
        s.index.name = 'kappa'
        return s

    def fit(self):
        logp = self.logp_k
        best = np.arange(self.n_kappa)[np.argmax(logp)]
        self.k.value = best

    def graph(self, pth):
        fig_name = pth.namebase
        fig_fmt = pth.ext.lstrip('.')
        fig_dir = pth.dirname()

        if not fig_dir.exists():
            fig_dir.makedirs_p()

        pymc.graph.graph(
            pymc.Model(self),
            name=fig_name,
            format=fig_fmt,
            path=fig_dir)

        return IPython.display.Image(pth)

    def plot_J_pdf(self, t, ax=None):
        p_fall = self.p_fall[t]
        J = self.J[t]
        X = np.arange(7)
        pX = np.exp([binomial_like(x, 6, p_fall) for x in X])

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(X+1, pX, ls='-', marker='o')
        ax.vlines(J.value+1, 0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlim(0.5, 7.5)
