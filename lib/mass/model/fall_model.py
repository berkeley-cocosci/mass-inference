import pymc
import numpy as np
import IPython
from pymc.distributions import normal_like

from .util import LazyProperty


class FallModel(object):

    # names of the keys that get stored when the model is pickled
    _state_keys = [
        'trials',
        'n_trial',
        'ipe_mean',
        'ipe_var',
        'exp',
        'kappas',
        'n_kappa',
        'prior',
        'pymc_values'
    ]

    def __init__(self, ipe, exp, trials, prior=None):
        self.trials = map(str, trials)
        self.n_trial = len(self.trials)

        self.ipe_mean = ipe.P_fall_mean.ix[self.trials]
        self.ipe_var = ipe.P_fall_var.ix[self.trials]
        self.exp = exp.ix[self.trials]

        self.kappas = map(float, self.ipe_mean.columns)
        self.n_kappa = len(self.kappas)

        if prior is None:
            self.prior = np.ones(self.n_kappa) / self.n_kappa
        elif len(prior) != self.n_kappa:
            raise ValueError("given prior is not the right shape")
        else:
            self.prior = np.array(prior)

        self.init_pymc_variables()

    def init_pymc_variables(self):
        self.variables = [self.S, self.k, self.p_fall, self.J]

    @property
    def pymc_values(self):
        # we don't need S because it is deterministic or J because it
        # is observed
        values = {}
        values['k'] = self.k.value
        values['p_fall'] = np.empty(self.p_fall.shape, dtype=np.float64)
        for i, p_fall in enumerate(self.p_fall):
            values['p_fall'][i] = p_fall.value
        return values

    @pymc_values.setter
    def pymc_values(self, values):
        self.k.value = values['k']
        for i, p_fall in enumerate(self.p_fall):
            p_fall.value = values['p_fall'][i]

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
        return pymc.Categorical('k', self.prior)

    @LazyProperty
    def p_fall(self):
        """Probability that the tower falls, independent of mass"""
        p_fall = np.empty(self.n_trial, dtype=object)
        k = self.k
        for t, S in enumerate(self.S):
            @pymc.stochastic(name="p_fall_%d" % t)
            def p_fall_t(value=0.5, k=k, S=S):
                mu = self.ipe_mean.ix[S, int(k)]
                tau = 1. / self.ipe_var.ix[S, int(k)]
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

    def fit(self):
        m = pymc.MAP(self)
        m.fit()
        assert self.k.value == m.k.value

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
