__all__ = [
    'Beta', 'Binomial', 'Dirichlet', 'Exponential', 'Gamma', 'Gaussian',
    'Multinomial', 'Uniform', 'E'
    ]
           
import scipy.stats.distributions as dists
import numpy.random as nprand

from scipy.special import betaln, gammaln, binom, iv, betainc, digamma
from scipy import integrate
from numpy import exp, array, resize, broadcast, log, sum, empty, nan, where
from numpy import concatenate, pi, ones, zeros, inf, cos, sin, empty_like
from numpy import max, min, sqrt, unique, floor, isnan

import circ

def bc(*args):
    bc = broadcast(*args)
    shape = bc.shape
    size = bc.size
    return shape, size


class RandomVariable(object):

    def __call__(self, size=None):
        return self.sample(size=size)

    def calcSupport(self):
        raise NotImplementedError
    def calcMean(self):
        raise NotImplementedError
    def calcMode(self):
        raise NotImplementedError
    def calcVar(self):
        raise NotImplementedError
    def calcSkewness(self):
        raise NotImplementedError

    @property
    def support(self):
        try:
            self._support
        except AttributeError:
            self._support = self.calcSupport()
        return self._support

    @property
    def mean(self):
        try:
            self._mean
        except AttributeError:
            self._mean = self.calcMean()
        return self._mean

    @property
    def mode(self):
        try:
            self._mode
        except AttributeError:
            self._mode = self.calcMode()
        return self._mode

    @property
    def var(self):
        try:
            self._var
        except AttributeError:
            self._var = self.calcVariance()
        return self._var

    @property
    def skewness(self):
        try:
            self._skewness
        except AttributeError:
            self._skewness = self.calcSkewness()
        return self._skewness

class ContinuousRandomVariable(RandomVariable):

    continuous = True
    discrete = False

    def pdf(self, x):
        prob = exp(self.logpdf(x))
        return prob

class DiscreteRandomVariable(RandomVariable):

    continuous = False
    discrete = True

    def PMF(self, x):
        prob = exp(self.logPMF(x))
        return prob

class ExpectedValue(object):
    def __getitem__(self, obj):
        try:
            return obj.mean
        except:
            raise ValueError, "not a random variable"
E = ExpectedValue()

################################################################################
class Beta(ContinuousRandomVariable):

    name = 'beta'

    def __init__(self, a, b):

        # a and b parameters
        self.a = array(a, dtype='f8', copy=True)
        self.b = array(b, dtype='f8', copy=True)

        if (self.a <= 0).any():
            raise ValueError, "invalid a"
        if (self.b <= 0).any():
            raise ValueError, "invalid b"

        self.shape, self.size = bc(self.a, self.b)
        self.a = resize(self.a, self.shape)
        self.b = resize(self.b, self.shape)

        # constant in the logpdf (Beta function)
        self.__const = betaln(self.a, self.b)


    def logPDF(self, x):
        a, b, C = self.a, self.b, self.__const
        y = array(x, dtype='f8')
        logprob = log(y)*(a-1) + log(1-y)*(b-1) - C
        return logprob
    def CDF(self, x):
        y = array(x, dtype='f8')
        cdf = betainc(self.a, self.b, y)
        return cdf


    def sample(self, size=None):
        samps = nprand.beta(self.a, self.b, size=size)
        return samps


    def calcSupport(self):
        return (0.0, 1.0)
    def calcMean(self):
        a, b = self.a, self.b
        mean = a / (a + b)
        return mean
    def calcMode(self):
        a, b = self.a, self.b
        mode = empty(a.shape, dtype='f8')
        idx = (a>1) & (b>1)
        mode[idx] = (a - 1.)[idx] / (a + b - 2.)[idx]
        mode[~idx] = nan
        return mode
    def calcVariance(self):
        a, b = self.a, self.b
        var = a*b / ((a+b)**2 * (a+b+1))
        return var
    def calcSkewness(self):
        a, b = self.a, self.b
        num = 2*(b-a) * sqrt(a+b+1)
        den = (a+b+2) * sqrt(a*b)
        skew = num / den
        return skew


    def __bernoulli(self, x, axis):
        a, b = self.a, self.b
        y = array(x, dtype='f8')
        total = sum(y, axis=axis)
        newa = a + total
        newb = b + y.shape[axis] - total
        return newa, newb
    def updateBernoulli(self, x, axis=0):
        newa, newb = self.__bernoulli(x, axis)
        posterior = Beta(newa, newb)
        return posterior
    def predictiveBernoulli(self, x, axis=0):
        newa, newb = self.__bernoulli(x, axis)
        predictive = Bernoulli(newa / (newa + newb))
        return predictive


    def __binomial(self, x, n, axis):
        a, b = self.a, self.b
        y = array(x, dtype='f8')
        total = sum(y, axis=axis)
        N = sum(n, axis=axis)
        newa = a + total
        newb = b + N - total
        return newa, newb
    def updateBinomial(self, x, n, axis=0):
        newa, newb = self.__binomial(x, n, axis)
        posterior = Beta(newa, newb)
        return posterior
    def predictiveBinomial(self, x, n, axis=0):
        newa, newb = self.__binomial(x, n, axis)
        predictive = BetaBinomial(newa, newb, n)
        return predictive


    def __geometric(self, x, axis):
        a, b = self.a, self.b
        y = array(x, dtype='f8')
        n = y.shape[axis]
        newa = a + n
        newb = b + sum(y, axis=axis)
        return newa, newb
    def updateGeometric(self, x, axis=0):
        newa, newb = self.__geometric(x, axis)
        posterior = Beta(newa, newb)
        return posterior
    
        
################################################################################
class BetaBinomial(DiscreteRandomVariable):

    name = 'beta-binomial'

    def __init__(self, a, b, n):

        self.a = array(a, dtype='f8', copy=True)
        self.b = array(b, dtype='f8', copy=True)
        self.n = array(n, dtype='i8', copy=True)

        if (self.a <= 0).any():
            raise ValueError, "invalid a"
        if (self.b <= 0).any():
            raise ValueError, "invalid b"

        self.shape, self.size = bc(self.a, self.b, self.n)
        self.a = resize(self.a, self.shape)
        self.b = resize(self.b, self.shape)
        self.n = resize(self.n, self.shape)

        # constant in the logpdf (Beta function)
        self.__const = betaln(self.a, self.b)

    def logPDF(self, x):
        a, b, n, C = self.a, self.b, self.n, self.__const
        k = array(k, dtype='i8')
        logprob = log(binom(n, k)) + betaln(k+a, n-k+b) - C
        return logprob
    def CDF(self, x):
        raise NotImplementedError


    def sample(self, size=None):
        a, b, n = self.a, self.b, self.n
        p = Beta(a, b)(size)
        k = Binomial(n, p)(size)
        return k
    

    def calcSupport(self):
        n = unique(self.n)
        if n.size > 1:
            raise AttributeError(
                "cannot calculate support for multidimensional "
                "beta-binomial distribution")
        support = tuple(range(n[0]+1))
        return support
    def calcMean(self):
        n, a, b = self.n, self.a, self.b
        mean = n*a / (a + b)
        return mean
    def calcVariance(self):
        n, a, b = self.n, self.a, self.b
        var = (n*a*b * (n+a+b)) / ((a+b)**2 * (a+b+1))
        return var
    def calcSkewness(self):
        n, a, b = self.n, self.a, self.b
        skew = ((a+b+2*n) * (b-a) * sqrt((1+a+b) / (n*a*b*(n+a+b)))) / (a+b+2)
        return skew


    def __hypergeometric(self, x, axis):
        raise NotImplementedError
    def updateHypergeometric(self, x, axis=0):
        newa, newb = self.__hypergeometric(x, axis)
        posterior = BetaBinomial(newa, newb, self.n)
        return posterior

        
################################################################################
class Bernoulli(RandomVariable):

    name = 'bernoulli'

    def __init__(self, p):

        self.p = array(p, dtype='f8', copy=True)
        
        if (p <= 0).any() or (p >= 1).any():
            raise ValueError("invalid value for p")

        self.shape, self.size = self.p.shape, self.p.size
        self.p = resize(self.p, self.shape)

    def logPMF(self, x):
        prob = log(self.PMF(x))
        return prob
    def PMF(self, x):
        y = array(x, dtype='f8', copy=True)
        y[y < 0] = nan
        y[y > 1] = nan
        prob = where(y, self.p, 1-self.p)
        return prob
    def CDF(self, x):
        raise NotImplementedError


    def sample(self, size=None):
        vals = nprand.uniform(0, 1, size=size) <= self.p
        return vals

    def calcSupport(self):
        return (0.0, 1.0)
    def calcMean(self):
        return self.p.copy()
    def calcMode(self):
        p = self.p
        mode = empty(self.shape, dtype='i8')
        mode[p<0.5] = 0
        mode[p>0.5] = 1
        mode[p==0.5] = nan
        return mode
    def calcVariance(self):
        p = self.p
        var = p * (1-p)
        return var
    def calcSkewness(self):
        p, q = self.p, 1-self.p
        skew = (q-p) / sqrt(q*p)
        return skew
    
        
################################################################################
class Binomial(DiscreteRandomVariable):

    name = 'binomial'

    def __init__(self, n, p):

        self.n = array(n, dtype='i8', copy=True)
        self.p = array(p, dtype='f8', copy=True)

        if (self.n <= 0).any():
            raise ValueError, "invalid n"
        if (self.p < 0).any() or (self.p > 1).any():
            raise ValueError, "invalid probability"

        self.shape, self.size = bc(self.n, self.p)
        self.n = resize(self.n, self.shape)
        self.p = resize(self.p, self.shape)


    def logPMF(self, x):
        n, p = self.n, self.p
        y = array(x, dtype='i8')
        C = log(binom(n, x))
        logprob = C + y*log(p) + (n-y)*log(1-p)
        return logprob
    def PMF(self, x):
        prob = exp(self.logPMF(x))
        return prob
    def CDF(self, x):
        raise NotImplementedError


    def sample(self, size=None):
        samps = nprand.binomial(self.n, self.p, size=size)
        return samps


    def calcSupport(self):
        n = unique(self.n)
        if n.size > 1:
            raise AttributeError(
                "cannot calculate support for multidimensional "
                "beta-binomial distribution")
        support = tuple(range(n[0]+1))
        return support
    def calcMean(self):
        n, p = self.n, self.p
        mean = n*p
        return mean
    def calcMode(self):
        n, p = self.n, self.p
        val = (n+1) * p
        isint = val.astype('i8')==val
        mode = empty(self.shape, dtype='f8')
        mode.fill(nan)
        mode[val==(n+1)] = n[val==(n+1)]
        mode[(val==0) | ~isint] = floor(val[val==0 | ~isint])
        return mode
    def calcVariance(self):
        n, p = self.n, self.p
        var = n*p*(1-p)
        return var
    def calcSkewness(self):
        n, p = self.n, self.p
        skew = (1-2*p) / sqrt(n*p*(1-p))
        return skew


################################################################################
class Categorical(DiscreteRandomVariable):

    def __init__(self, k, p, axis=-1):

        self.k = array(n, dtype='i8', copy=True)
        self.p = array(p, dtype='f8', copy=True)
        self.axis = axis

        if (self.k <= 0).any():
            raise ValueError("invalid number of categories")
        if (sum(p, axis=self.axis) != 1).any():
            raise ValueError("probabilities do not sum to 1")

        self.shape, self.size = bc(self.k, self.p)
        self.k = resize(self.k, self.shape)
        self.p = resize(self.p, self.shape)
    

################################################################################
class Dirichlet(ContinuousRandomVariable):

    def __init__(self, alpha, axis=-1):

        self.alpha = array(alpha, dtype='f8', copy=True)
        self.axis = int(axis)

        if (self.alpha <= 0).any():
            raise ValueError, "invalid alpha value"
        if self.alpha.shape[self.axis] < 2:
            raise ValueError, "dimensions of simplex must be >= 2"

        self.shape, self.size = self.alpha.shape, self.alpha.size
        self.alpha = resize(self.alpha, self.shape)
        self.gamma = Gamma(self.alpha, 1) # for sampling
        
        # newdim = range(self.alpha.ndim)
        # newdim.append(newdim.remove(axis % self.alpha.ndim))
        # self._alpha = self.alpha.transpose(newdim).reshape(
        #     (-1, self.shape[axis]))

        # coefficient for logPDF
        self.__const = sum(gammaln(alpha), axis=self.axis)
        self.__const -= gammaln(sum(alpha, axis=self.axis))

    def logPDF(self, x):
        alpha, axis, C = self.alpha, self.axis, self.__const
        y = array(x, dtype='f8')
        logprob = sum(log(y) * (alpha-1), axis=axis) - C
        return logprob
    def CDF(self, x):
        raise NotImplementedError


    def sample(self, size=None):
        samps = self.gamma.sample(size=size)
        total = sum(samps, axis=self.axis)
        sl = [slice(None) for x in self.shape]
        sl[axis] = None
        samps /= total[sl]
        return samps


    def calcSupport(self):
        pass
    def calcMean(self):
        alpha, axis = self.alpha, self.axis
        sl = [slice(None) for x in self.shape]
        sl[axis] = None
        mean = alpha / sum(alpha, axis=axis)[sl]
        return mean
    def calcMode(self):
        alpha, axis = self.alpha, self.axis
        sl = [slice(None) for x in self.shape]
        sl[axis] = None
        mode = (alpha - 1) / (sum(alpha, axis=axis)[sl] - self.shape[axis])
        mode[alpha<=1] = nan
        return mode
    def calcVariance(self):
        pass
    def calcSkewness(self):
        pass


    def __multinomial(self, x, axis):
        pass
    def updateMultinomial(self, x, axis=0):
        y = array(x, dtype='i8')
        newalpha = self.alpha + sum(y, axis=axis)
        newdist = Dirichlet(newalpha, self.axis)
        return newdist
    def predictiveMultinomial(self, x, axis=0):
        pass


    def __categorical(self, x, axis):
        pass
    def updateCategorical(self, x, axis=0):
        pass
    def predictiveCategorical(self, x, axis=0):
        pass

################################################################################
class Exponential(ContinuousRandomVariable):

    def __init__(self, lam):

        self.lam = array(lam, dtype='f8', copy=True)
        self.loglam = log(self.lam)
        self.beta = 1.0 / self.lam

    def logPDF(self, x):
        lam, loglam = self.lam, self.loglam
        y = array(x, dtype='f8')
        logprob = loglam - lam*y
        return logprob

    def pdf(self, x):
        """
        http://en.wikipedia.org/wiki/Exponential_distribution
        """

        prob = exp(self.logPDF(x))
        return prob

    def sample(self, size=None):
        samps = nprand.exponential(self.beta, size=size)
        return samps

################################################################################
class Gamma(ContinuousRandomVariable):

    def __init__(self, k, theta):

        self.k = array(k, dtype='f8', copy=True)
        self.theta = array(theta, dtype='f8', copy=True)

        # constant for the logPDF
        self.__const = gammaln(self.k) + self.k*log(self.theta)

    def logPDF(self, x):
        k, theta, C = self.k, self.theta, self.__const
        y = array(x, dtype='f8')
        logprob = (k-1)*log(y) - y/theta - C
        return logprob

    def pdf(self, x):
        """
        http://en.wikipedia.org/wiki/Gamma_distribution
        """

        prob = exp(self.logPDF(x))
        return prob

    def sample(self, size=None):
        samps = nprand.gamma(shape=self.k, scale=self.theta, size=size)
        return samps

    def cp_poisson(self, x, axis=0):
        k, theta = self.k, self.theta
        y = array(x, dtype='f8')
        newk = k + sum(y, axis=axis)
        newtheta = theta / (y.shape[axis]*theta + 1)
        posterior = Gamma(newk, newtheta)
        return posterior

    def cp_gaussian(self, x, mu, axis=0):
        a, b = self.k, 1./self.theta
        y = array(x, dtype='f8')
        n = y.shape[axis]
        newa = a + n/2.
        newb = b + sum((y - array(mu))**2, axis=axis) / 2.
        newk = newa
        newtheta = 1. / newb
        posterior = Gamma(newk, newtheta)
        return posterior

    def cp_exponential(self, x, axis=0):
        a, b = self.k, 1./self.theta
        y = array(x, dtype='f8')
        n = y.shape[axis]
        newa = a + n
        newb = b + sum(y, axis=axis)
        newk = newa
        newtheta = 1. / newb
        posterior = Gamma(newk, newtheta)
        return posterior
        

################################################################################
class Gaussian(ContinuousRandomVariable):

    def __init__(self, mu, sigma):

        self.mu = array(mu, dtype='f8', copy=True)
        self.sigma = array(sigma, dtype='f8', copy=True)
        self.sigma2 = array(sigma)**2

        # constant in the logPDF
        self.__const = -0.5 * (log(2*pi) + log(self.sigma2))

    def logPDF(self, x):
        mu, sigma2, C = self.mu, self.sigma2, self.__const
        y = array(x, dtype='f8')
        logprob = C - ((y - mu)**2 / (2*sigma2))
        return logprob

    def pdf(self, x):
        """
        http://en.wikipedia.org/wiki/Normal_distribution
        """
        
        prob = exp(self.logPDF(x))
        return prob

    def sample(self, size=None):
        samps = nprand.normal(loc=self.mu, scale=self.sigma, size=size)
        return samps

################################################################################
class InvGamma(ContinuousRandomVariable):
    pass

################################################################################
class Multinomial(DiscreteRandomVariable):

    def __init__(self, n, p, axis=-1):

        self.n = array(n, dtype='i8', copy=True)
        self.p = array(p, dtype='f8', copy=True)
        self.axis = axis

        if (self.n <= 0).any():
            raise ValueError("invalid n")
        if (abs(sum(p, axis=self.axis) - 1) > 1e-6).any():
            raise ValueError("probabilities do not sum to 1")

        self.shape, self.size = bc(self.n, self.p)
        self.nshape = list(self.shape)[:]
        self.nshape.pop(self.axis)
        self.nshape = tuple(self.nshape)
        self.n = resize(self.n, self.nshape)
        self.p = resize(self.p, self.shape)

        self.__const = gammaln(self.n+1)

    def logPMF(self, x):
        p, axis, C = self.p, self.axis, self.__const
        y = array(x, dtype='i8')
        logprob = sum(y*log(p), axis=axis) - sum(gammaln(y+1), axis=axis) - C
        logprob[isnan(logprob)] = -inf
        return logprob


    # def sample(self, size=None):
    #     n, p = self.n, self.p
    #     samps = nprand.multinomial(

################################################################################
class MVGaussian(ContinuousRandomVariable):
    pass

################################################################################
class Poisson(DiscreteRandomVariable):
    pass

################################################################################
class Uniform(ContinuousRandomVariable):

    def __init__(self, low, high):

        self.low = array(low, dtype='f8', copy=True)
        self.high = array(high, dtype='f8', copy=True)

    def logPDF(self, x):
        low, high = self.low, self.high
        logprob = ones(x.shape) / (high - low)
        logprob[x < low] = -inf
        logprob[x >= high] = -inf
        return logprob

    def pdf(self, x):
        prob = exp(self.logPDF(x))
        return prob

    def sample(self, size=None):
        samps = nprand.uniform(low=low, high=high, size=size)
        return samps

################################################################################
class VonMises(ContinuousRandomVariable):

    def __init__(self, theta, kappa):

        self.theta = array(theta, dtype='f8', copy=True)
        self.kappa = array(kappa, dtype='f8', copy=True)

        # constant for logPDF
        self.__const = -log(2 * pi * iv(0, self.kappa))

    def logPDF(self, x):
        theta, kappa, C = self.theta, self.kappa, self.__const
        y = array(x, dtype='f8')
        logprob = C + (kappa * cos(y - theta))
        return logprob

    def pdf(self, x):
        """Computes the circular Von Mises PDF with preferred direction
        self.theta and concentration self.kappa at each of the angles in 'x'.

        The vmpdf is given by f(phi) =
        (1/(2pi*I0(kappa))*exp(kappa*cos(phi-theta)

        Parameters
        ----------
        x : ndarray
            The array of angles

        Returns
        -------
        out : ndarray
            The Von Mises PDF for 'x'

        References
        ----------
        http://www.jstatsoft.org/v31/i10
        
        """

        prob = exp(self.logPDF(x))
        return prob

    @staticmethod
    def MAP(x, axis=None, nanrobust=False):
        alpha = array(x, dtype='f8')
        theta = VonMises.MAP_theta(alpha, axis=axis, nanrobust=nanrobust)
        kappa = VonMises.MAP_kappa(alpha, axis=axis, nanrobust=nanrobust)
        dist = VonMises(theta, kappa)
        return dist

    @staticmethod
    def MAP_theta(x, axis=None, nanrobust=False):
        alpha = array(x, dtype='f8')
        if nanrobust:
            theta = circ.nanmean(alpha, axis=axis)
        else:
            theta = circ.mean(alpha, axis=axis)
        return theta

    @staticmethod
    def MAP_kappa(x, axis=None, nanrobust=False):
        alpha = array(x, dtype='f8')
        n = alpha.size if axis is None else alpha.shape[axis]

        if nanrobust:
            R = circ.nanresvec(alpha, axis=axis)
        else:
            R = circ.resvec(alpha, axis=axis)

        i1 = R < 0.53
        i2 = (R>=0.53) & (R<0.85)
        i3 = ~(i1 | i2)

        kappa = empty_like(R) # allocate
        kappa[i1] = 2*R[i1] + R[i1]**3 + 5*R[i1]**5/6
        kappa[i2] = -.4 + 1.39*R[i2] + 0.43/(1-R[i2])
        kappa[i3] = 1/(R[i3]**3 - 4*R[i3]**2 + 3*R[i3])
        
        if (n<15) & (n>1):
            i1 = kappa < 2
            i2 = ~i1
            kappa[i1] = max(concatenate(
                [kappa[i1]-2*(n*kappa[i1])**-1,
                 zeros(i1.shape)],
                axis=i1.ndim-1), axis=i1.ndim-1)
            kappa[i2] = (n-1)**3*kappa[i2]/(n**3+n)

        return kappa

################################################################################
class Wishart(ContinuousRandomVariable):
    pass
