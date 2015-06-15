__all__ = ['percent_fell', 'more_than_half_fell', 'more_than_one_fell']


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


def more_than_one_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    answer = (samps > 1).astype(int)
    return answer
