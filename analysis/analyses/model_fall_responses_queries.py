__all__ = ['percent_fell', 'more_than_half_fell', 'at_least_one_fell']


def percent_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    answer = samps / 10.0
    return answer


def more_than_half_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    answer = (samps > 5).astype(int)
    return answer


def at_least_one_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    answer = (samps > 0).astype(int)
    return answer
