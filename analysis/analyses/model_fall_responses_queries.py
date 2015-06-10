import util
import numpy as np

__all__ = ['percent_fell', 'more_than_half_fell', 'more_than_one_fell']


def percent_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    answer = (samps / 10.0).apply(util.bootstrap_mean).T
    answer['stddev'] = np.std(samps / 10.0)
    answer['mean'] = (samps / 10.0).mean()
    return answer


def more_than_half_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    answer = (samps > 5).apply(util.beta).T
    answer['stddev'] = util.beta_stddev(samps > 5)
    answer['mean'] = util.beta_mean(samps > 5)
    return answer


def more_than_one_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    answer = (samps > 1).apply(util.beta).T
    answer['stddev'] = util.beta_stddev(samps > 1)
    answer['mean'] = util.beta_mean(samps > 1)
    return answer
