import pymc
import numpy as np
from pymc.distributions import normal_like
#import matplotlib.pyplot as plt

from .util import LazyProperty


class FallModel(object):
    def __init__(self, ipe, exp, trials, prior=None):
        self.ipe = ipe
        self.exp = exp.copy()

        self.trials = map(str, trials)
        self.n_trial = len(self.trials)

        self.kappas = map(float, self.ipe.P_fall_mean.columns)
        self.n_kappa = len(self.kappas)

        if prior is None:
            self.prior = np.ones(self.n_kappa) / self.n_kappa
        elif len(prior) != self.n_kappa:
            raise ValueError("given prior is not the right shape")
        else:
            self.prior = np.array(prior)

        # if sorted(self.ipe.P_fall_mean.index) != sorted(self.trials):
        #     print self.ipe.P_fall_mean.index
        #     print self.trials
        #     raise ValueError("IPE index does not match trials")

        # if sorted(self.exp.index) != sorted(self.trials):
        #     print self.exp.index
        #     print self.trials
        #     raise ValueError("Experiment index does not match trials")

        self.variables = [self.S, self.k, self.p_fall, self.J]

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
        return pymc.Categorical('k', self.prior)

    @LazyProperty
    def p_fall(self):
        """Probability that the tower falls, independent of mass"""
        p_fall = np.empty(self.n_trial, dtype=object)
        k = self.k
        for t, S in enumerate(self.S):
            @pymc.stochastic(name="p_fall_%d" % t)
            def p_fall_t(value=0.5, k=k, S=S):
                mu = self.ipe.P_fall_mean.ix[S, int(k)]
                tau = 1. / self.ipe.P_fall_var.ix[S, int(k)]
                return normal_like(value, mu, tau)
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
