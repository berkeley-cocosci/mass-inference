import pymc
import numpy as np
from snippets.safemath import normalize
from pymc.distributions import bernoulli_like
import matplotlib.pyplot as plt

from .util import LazyProperty


class ObserverModel(object):

    # names of the keys that get stored when the model is pickled
    _state_keys = [
        'trials',
        'n_trial',
        'ipe',
        'fb',
        'kappa0',
        'kappas',
        'n_kappa',
        'prior',
        'pymc_values'
    ]

    def __init__(self, ipe, fb, trials, kappa0, prior=None):
        self.trials = map(str, trials)
        self.n_trial = len(self.trials)

        self.ipe = ipe.P_fall_smooth.ix[self.trials]
        self.fb = fb.fall[kappa0].ix[self.trials]
        self.kappa0 = kappa0

        self.kappas = map(float, self.ipe.columns)
        self.n_kappa = len(self.kappas)

        if prior is None:
            self.prior = np.log(np.ones(self.n_kappa) / self.n_kappa)
        elif len(prior) != self.n_kappa:
            raise ValueError("given prior is not the right shape")
        else:
            self.prior = np.log(np.array(prior))

        self.init_pymc_variables()

    def init_pymc_variables(self):
        self.variables = [self.J_fall, self.J_mass, self.F, self.k]

    @property
    def pymc_values(self):
        values = {}

        values['J_fall'] = np.empty(self.J_fall.shape, dtype=np.float64)
        for i, J_fall in enumerate(self.J_fall):
            values['J_fall'][i] = J_fall.value

        values['J_mass'] = np.empty(self.J_mass.shape, dtype=np.float64)
        for i, J_mass in enumerate(self.J_mass):
            values['J_mass'][i] = J_mass.value

        values['F'] = np.empty(self.F.shape, dtype=np.float64)
        for i, F in enumerate(self.F):
            values['F'][i] = F.value

        values['k'] = np.empty(self.k.shape, dtype=np.float64)
        for i, k in enumerate(self.k):
            values['k'][i] = k.value

        return values

    @pymc_values.setter
    def pymc_values(self, values):
        for i, J_fall in enumerate(self.J_fall):
            J_fall.value = values['J_fall'][i]

        for i, J_mass in enumerate(self.J_mass):
            J_mass.value = values['J_mass'][i]

        for i, F in enumerate(self.F):
            F.value = values['F'][i]

        for i, k in enumerate(self.k):
            k.value = values['k'][i]

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
        return np.array(self.trials)

    @LazyProperty
    def B(self):
        """Belief about mass ratio"""
        B = np.empty((self.n_trial + 1, self.n_kappa), dtype=float)
        B[0] = self.prior
        for t, S in enumerate(self.S):
            F = self.fb[S]
            if np.isnan(F):
                B[t+1] = B[t]
            else:
                lh = [bernoulli_like(F, p) for p in self.ipe.ix[S]]
                B[t+1] = normalize(B[t] + lh)[1]
        return B

    @LazyProperty
    def p_fall(self):
        """Probability that the tower falls, independent of mass"""
        p_fall = np.empty(self.n_trial)
        for t, S in enumerate(self.S):
            p_fall[t] = np.log((self.ipe.ix[S] * np.exp(self.B[t])).sum())
        return p_fall

    @LazyProperty
    def p_mass(self):
        """Probability that kappa > 0"""
        p_mass = np.empty(self.n_trial + 1)
        ibig = np.array(self.kappas) > 0
        ieq = np.array(self.kappas) == 0
        for t in xrange(self.n_trial + 1):
            B = np.exp(self.B[t])
            if not np.allclose(B.sum(), 1):
                print B
                print B.sum()
                raise ValueError
            p_mass[t] = np.log(B[ibig].sum() + 0.5 * B[ieq])
        return p_mass

    @LazyProperty
    def J_fall(self):
        """Bernoulli judgment of whether the tower will fall"""
        J = np.empty(self.n_trial, dtype=object)
        for t in xrange(self.n_trial):
            J[t] = pymc.Bernoulli('J_fall_%d' % t, np.exp(self.p_fall[t]))
        return J

    @LazyProperty
    def J_mass(self):
        """Bernoulli judgment of whether kappa > 0"""
        J = np.empty(self.n_trial + 1, dtype=object)
        for t in xrange(self.n_trial + 1):
            J[t] = pymc.Bernoulli('J_mass_%d' % t, np.exp(self.p_mass[t]))
        return J

    @LazyProperty
    def F(self):
        """Bernoulli feedback"""
        F = np.empty(self.n_trial, dtype=object)
        for t, S in enumerate(self.S):
            ipe = self.ipe.ix[S]
            k = self.k[t]
            value = self.fb[S]

            if np.isnan(value):
                @pymc.stochastic(dtype=int, observed=False, name="F_%d" % t)
                def F_t(value=1, k=k):
                    return bernoulli_like(value, ipe[int(k)])

            else:
                @pymc.stochastic(dtype=int, observed=True, name="F_%d" % t)
                def F_t(value=value, k=k):
                    return bernoulli_like(value, ipe[int(k)])

            F[t] = F_t

        return F

    @LazyProperty
    def k(self):
        """Kappa"""
        k = np.empty(self.n_trial + 1, dtype=object)
        for t in xrange(self.n_trial + 1):
            k[t] = pymc.Categorical('k_%d' % t, np.exp(self.B[t]))
        return k

    def sample_k(self, n=1, T=None):
        """Sample `n` values of kappa weighted by the belief about mass ratio
        at time `T`.

        """
        # parameter checking
        if T is None:
            T = range(self.n_trial + 1)
        elif isinstance(T, int):
            T = [T]

        # generate the random values
        k = np.empty((len(T), n))
        for i, t in enumerate(T):
            for j in xrange(n):
                k[i, j] = self.k[t].rand()

        return k.squeeze()

    def sample_J_fall(self, n=1, T=None):
        """Sample `n` judgments of whether the tower(s) at time `T` will
        fall.

        """
        # parameter checking
        if T is None:
            T = range(self.n_trial)
        elif isinstance(T, int):
            T = [T]

        # generate the judgments
        J = np.empty((len(T), n))
        for i, t in enumerate(T):
            for j in xrange(n):
                J[i, j] = self.J_fall[t].rand()

        return J.squeeze()

    def sample_J_mass(self, n=1, T=None):
        """Sample `n` judgments of whether kappa is greater than 0, based on
        the belief at time(s) `T`.

        """
        # parameter checking
        if T is None:
            T = range(self.n_trial)
        elif isinstance(T, int):
            T = [T]

        # generate the judgments
        J = np.empty((len(T), n))
        for i, t in enumerate(T):
            for j in xrange(n):
                J[i, j] = self.J_mass[t+1].rand()

        return J.squeeze()

    def eval_J_fall(self, J_fall):
        """Evaluate the likelihood of fall judgments `J_fall`."""
        # parameter checking
        if J_fall.ndim != 1:
            raise ValueError("J_fall may only have one dimension")

        # evaluate
        p_J_fall = np.empty(J_fall.shape)
        for t in xrange(J_fall.size):
            self.J_fall[t].set_value(J_fall[t])
            p_J_fall[t] = self.J_fall[t].logp

        return p_J_fall

    def eval_J_mass(self, J_mass):
        """Evaluate the likelihood of mass judgments `J_mass`."""
        # parameter checking
        if J_mass.ndim != 1:
            raise ValueError("J_mass may only have one dimension")

        # evaluate
        p_J_mass = np.empty(J_mass.shape) * np.nan
        for t in xrange(J_mass.size):
            if np.isnan(J_mass[t].value):
                continue
            self.J_mass[t].set_value(J_mass[t])
            p_J_mass[t] = self.J_mass[t].logp

        return p_J_mass

    def plot_belief(self, ax=None):
        """Plot a heatmap of belief about mass ratio over time."""

        # parameter checking
        if ax is None:
            fig, ax = plt.subplots()

        im = ax.imshow(np.exp(self.B).T, cmap='gray', interpolation='nearest')
        ax.set_yticks([self.kappas.index(x) for x in [-1, 0, 1]])
        ax.set_yticklabels(['0.1', '1.0', '10'])
        ax.set_xlabel("Trial number")
        ax.set_ylabel("Mass ratio ($r$)")
        cb = plt.colorbar(im)
        cb.ax.set_title("$p(r)$")

    def plot_kappa(self, T, ax=None, n=10000):
        """Plot a histogram of sampled kappas and curve of belief at time(s)
        `T`.

        """

        # parameter checking
        if ax is None:
            fig, ax = plt.subplots()
        if isinstance(T, int):
            T = [T]

        ax.hist(
            self.sample_k(n, T).T,
            bins=self.n_kappa,
            normed=True)
        ax.legend(["t=%d" % t for t in T], loc=0)

        ax.plot(np.exp(self.B[T]).T, lw=2, color='k')

        ax.set_xticks([self.kappas.index(x) for x in [-1, 0, 1]])
        ax.set_xticklabels(['0.1', '1.0', '10'])
        ax.set_xlabel("Mass ratio ($r$)")
        ax.set_ylabel("$p(r)$")
