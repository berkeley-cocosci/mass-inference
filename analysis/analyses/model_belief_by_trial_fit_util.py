import numpy as np
from scipy.optimize import minimize_scalar as minimize


def log_laplace(x, mu=0, b=1):
    """Compute p(x | mu, b) according to a laplace distribution"""
    # (1 / 2*b) * np.exp(-np.abs(x - mu) / b)
    c = -np.log(2 * b)
    e = -np.abs(x - mu) / b
    return c + e


def make_posterior(X, y):
    """Returns a function that takes an argument for the hypothesis, and that
    then computes the posterior probability of that hypothesis given X and y

    """
    def f(B):
        # compute the prior
        log_prior = log_laplace(B, mu=1, b=1)

        # compute the likelihood
        p = 1.0 / (1 + np.exp(-(X * B)))
        log_lh = np.log((y * p) + ((1 - y) * (1 - p))).sum()

        # compute the posterior
        log_posterior = log_lh + log_prior

        return -log_posterior

    return f


def logistic_regression(X, y, verbose=False):
    """Performs a logistic regression with one coefficient for predictors X and
    output y.

    """
    log_posterior = make_posterior(X, y)
    res = minimize(log_posterior)
    if verbose:
        print res
    return float(res['x'])


def fit_responses(df, model_name, verbose=False):
    """Fits participant responses using a logistic regression. The given data
    frame should have, at least, columns for 'mass? response' and 'log_odds'. A
    new dataframe will be returned, minus columns for 'mass? response' and
    'log_odds', but with columns 'B' (the fitted parameter), 'p' (the fitted
    belief for r=10), and 'p correct' (the fitted probability of answering
    correctly).

    """
    counterfactual, version, kappa0, pid = df.name

    df2 = df.dropna()
    y = np.asarray(df2['mass? response'])
    X = np.asarray(df2['log_odds'])

    if model_name == 'chance':
        B = 0
    else:
        B = logistic_regression(X, y, verbose)

    f = np.asarray(df['log_odds']) * B
    f_raw = np.asarray(df['log_odds'])
    mu = 1.0 / (1 + np.exp(-f))
    mu_raw = 1.0 / (1 + np.exp(-f_raw))

    new_df = df.copy().drop(['mass? response', 'log_odds'], axis=1)
    new_df['B'] = B
    new_df['p'] = mu
    new_df['p raw'] = mu_raw

    if kappa0 < 0:
        new_df['p correct'] = 1 - mu
        new_df['p correct raw'] = 1 - mu_raw
    else:
        new_df['p correct'] = mu
        new_df['p correct raw'] = mu_raw

    return new_df
