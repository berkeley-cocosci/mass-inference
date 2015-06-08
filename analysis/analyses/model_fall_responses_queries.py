import util

__all__ = ['percent_fell', 'more_than_half_fell', 'more_than_one_fell']


def percent_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    answer = (samps / 10.0).apply(util.bootstrap_mean)
    return answer.T


def more_than_half_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    answer = (samps > 5).apply(util.beta)
    return answer.T


def more_than_one_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    answer = (samps > 1).apply(util.beta)
    return answer.T
