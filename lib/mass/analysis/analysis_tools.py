import pandas as pd
import numpy as np
import scipy

import matplotlib
import matplotlib.cm as cm

from statsmodels.stats.proportion import binom_test
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.proportion import proportions_chisquare

from mass.model import model_observer as mo

from snippets.safemath import normalize
from snippets import datapackage as dpkg

CONDITION = ['order', 'feedback', 'ratio']
PID = CONDITION + ['pid']
PARAMS = ['sigma', 'phi', 'kappa']

binom_mean = mo.binom_mean
binom_std = mo.binom_std


def uncounterbalance(arr):
    cb = arr.name
    if cb == 0:
        arr.replace({
            'mass? response': {
                'yellow': 10,
                'red': 0.1,
            }
        }, inplace=True)
    elif cb == 1:
        arr.replace({
            'mass? response': {
                'yellow': 0.1,
                'red': 10,
            }
        }, inplace=True)
    else:
        raise ValueError("cb is %s" % cb)
    return arr


def drop_invalid_pids(hdata):

    dd = (hdata[PID + ['code', 'worker_id', 'timestamp']]
          .drop_duplicates()
          .dropna(subset=['worker_id'])
          .sort('timestamp'))

    # get the list of all pids (+ conditions)
    all_pids = list(dd.dropna(subset=['pid', ])
                    .set_index(PID)
                    .index.unique())
    # get the list of only valid pids (+ conditions)
    good_pids = list(dd.drop_duplicates(cols=['worker_id'])
                     .dropna(subset=['pid', 'code'])
                     .set_index(PID)
                     .index.unique())
    # the difference is the bad pids
    bad_pids = set(all_pids) - set(good_pids)

    # drop the bad pids from the data
    data = (hdata
            .dropna(subset=['pid', 'worker_id'])
            .set_index(PID)
            .drop(bad_pids)
            .reset_index())

    bad_pids = pd.DataFrame(sorted(bad_pids))
    bad_pids.columns = PID
    print "%d invalid pids:" % len(bad_pids)
    print bad_pids.groupby('order').apply(len)
    print

    return data


def drop_failed_posttest(hdata, datapath):
    shdp = dpkg.DataPackage.load(datapath.joinpath(
        "model/stability_sameheight_fall.dpkg"))
    model = shdp.load_resource('model.csv')
    sh = (model
          .set_index(PARAMS + ['sample', 'stimulus'])
          .xs(0.0, level='sigma')
          .xs(0.0, level='phi')
          .xs(0.0, level='kappa')
          .xs(0, level='sample')
          .reset_index())

    pt = hdata.groupby('mode').get_group('posttest')
    mpt = (pd.merge(pt, sh, on='stimulus')
           .set_index(PID + ['stimulus'])
           .sort()
           [['fall? response', 'nfell']])

    correct = (mpt['fall? response'] == (mpt['nfell'] > 0))
    ncorrect = correct.groupby(level=PID).sum()
    failed = sorted(ncorrect.index[ncorrect < 5])

    data = hdata.set_index(PID).drop(failed).reset_index()

    failed = pd.DataFrame(failed)
    failed.columns = PID
    print "%d failed posttest:" % len(failed)
    print failed.groupby('order').apply(len)
    print

    return data


def get_condition_info(human_data):
    # extract trials and stimuli from the human data
    trialinfo = human_data[['trial', 'stimulus']].drop_duplicates()
    trialinfo.sort('trial', inplace=True)
    trials = np.array(trialinfo.trial)
    stims = np.array(trialinfo.stimulus)

    out = pd.Series(
        stims, index=trials, name='stimulus')
    out.index.name = 'trial'

    return out


def simulate_model(condinfo, ipe, fb, n_fake=2000):
    """Generate model stability responses, explicit mass judgments,
    and belief over time for each experimental condition in 'conds'.

    """

    # the condition to generate data for
    order, fbtype, ratio = condinfo.name

    # extract trials and stimuli from the human data
    trialinfo = condinfo.drop_duplicates()
    trials = np.array(trialinfo.trial)
    stims = np.array(trialinfo.stimulus)

    # random number generator
    rso = np.random.RandomState(1)

    # convert the data frames to numpy arrays
    ipe_mat = make_ipe_mat(ipe, stims)
    fb_mat = make_fb_mat(fb, stims, fbtype, ratio)

    # start with a uniform prior belief, regardless of condition
    kappas = ipe.index.get_level_values('kappa').unique().astype('f8')
    prior = make_uniform_prior(kappas)

    # simulate fall? responses and model belief over time
    responses, model_belief = mo.simulateResponses(
        n_fake, fb_mat, ipe_mat, kappas,
        prior=prior, p_ignore=0.0, smooth=True, rso=rso)

    # create dataframe for responses
    mdata = make_mdata(responses, trials, stims, fbtype)
    # create dataframe for model belief
    mbelief = make_mbelief(model_belief, trials, kappas)

    return mdata, mbelief


def make_ipe_mat(ipe, stims):
    """Extract the relevant data from ipe into a 3d numpy array, where the
    dimensions correspond to (stim, ratio, sample)

    """
    ipe_mat = np.array(map(
        lambda x: np.asarray(ipe[x].unstack('sample')),
        stims))
    return ipe_mat


def make_uniform_prior(kappas):
    prior = normalize(np.zeros(len(kappas)))[1]
    return prior


def make_fb_mat(fb, stims, fbtype, ratio):

    if fbtype == 'nfb':
        # no feedback condition, so just give nans
        fb_mat = np.empty(len(stims)) * np.nan

    elif fbtype in ('fb', 'vfb'):
        # binary feedback condition, so extract the relevant data from
        # fb into a 2d numpy array, where the dimensions correspond to
        # (stim, ratio)
        fb_mat = np.asarray(
            fb[stims]
            .groupby(level='kappa')
            .get_group(np.log10(float(ratio)))).squeeze()

    else:
        raise ValueError("invalid feedback type: %s" % fbtype)

    return fb_mat


def make_mdata(responses, trials, stims, fbtype):
    fall_responses, mass_responses = responses
    n_fake = fall_responses.shape[0]

    ix = pd.MultiIndex.from_tuples(
        zip(trials, stims),
        names=['trial', 'stimulus'])
    cols = pd.Index(
        np.arange(n_fake, dtype=int),
        name='pid')

    mdata = pd.DataFrame({
        'fall? response': pd.DataFrame(
            fall_responses.T,
            index=ix, columns=cols).stack(),
        'mass? response': pd.DataFrame(
            mass_responses.T,
            index=ix, columns=cols).stack()
    }).reset_index()

    if fbtype == 'nfb':
        mdata['mass? response'] = np.nan

    return mdata


def make_mbelief(belief_mat, trials, kappas):
    # create a dataframe for model belief as well
    mbelief = pd.DataFrame(
        belief_mat.T,
        index=kappas,
        columns=np.append([trials[0]-1], trials),
        dtype='f8')
    mbelief.index.name = 'kappa'
    mbelief.columns.name = 'trial'
    mbelief = mbelief.stack().reset_index().rename(columns={0: 'log_p'})
    return mbelief


def pfell_smoothed(samps):
    kappa = list(samps.index.get_level_values('kappa'))
    smoothed_mat = mo.IPE(
        np.ones(1), np.asarray(samps)[None],
        kappa, smooth=True)[0]
    smoothed = pd.Series(
        smoothed_mat, index=samps.index, dtype='f8')
    return smoothed


def stim_colors(stims):
    colors = cm.hsv(np.linspace(0.0, 0.86, len(stims)))
    return dict(zip(stims, colors))


def make_cmap(name, c1, c2, c3):
    """Make a cmap that fades from color c1 to color c3 through c2"""
    colors = {
        'red': (
            (0.0, c1[0], c1[0]),
            (0.50, c2[0], c2[0]),
            (1.0, c3[0], c3[0]),),
        'green': (
            (0.0, c1[1], c1[1]),
            (0.50, c2[1], c2[1]),
            (1.0, c3[1], c3[1]),),
        'blue': (
            (0.0, c1[2], c1[2]),
            (0.50, c2[2], c2[2]),
            (1.0, c3[2], c3[2]))}
    cmap = matplotlib.colors.LinearSegmentedColormap(name, colors, 1024)
    return cmap


def check_mass_responses(x):
    ratiostr = x.index.get_level_values('ratio').unique()
    ratio = float(np.array(map(float, ratiostr)))
    correct = (x == ratio).astype('f8')
    return correct


def get_correct_mass_responses(df, species):
    correct = (
        df.set_index(PID + ['trial'])['mass? response']
        .dropna()
        .groupby(level=CONDITION)
        .transform(check_mass_responses)
        .reset_index()
        .rename(columns={'mass? response': 'correct'})
    )
    correct['trial'] -= 5
    correct['species'] = species
    return correct


def chi2(x):
    k = x.sum()
    dof = k.size
    n = x.shape[0]
    chi2, p, (contingency, expected) = proportions_chisquare(k, n)
    out = pd.Series(
        [dof, n, chi2, p],
        index=['dof', 'n', 'chi2', 'p'])
    return out


def test_gt_chance(x):
    n = len(x)
    k = x.sum()
    p = binom_test(k, n, prop=0.5, alternative='larger')
    return p


def test_diff_model(x):
    groups = x.groupby(level='species')
    human = groups.get_group('human')
    model = groups.get_group('model')
    b = binom_test(
        human.sum(), len(human),
        prop=model.mean(), alternative='two-sided')
    return b


def test_gt_model(x):
    groups = x.groupby(level='species')
    human = groups.get_group('human')
    model = groups.get_group('model')
    b = binom_test(
        human.sum(), len(human),
        prop=model.mean(), alternative='larger')
    return b


def test_lt_model(x):
    groups = x.groupby(level='species')
    human = groups.get_group('human')
    model = groups.get_group('model')
    b = binom_test(
        human.sum(), len(human),
        prop=model.mean(), alternative='smaller')
    return b


def test_vis_gt_bin(x):
    groups = x.groupby(level='feedback')
    binary = groups.get_group('fb')
    visual = groups.get_group('vfb')

    kb = binary.sum()
    nb = len(binary)
    kv = visual.sum()
    nv = len(visual)

    ab = 1 + kb
    bb = 1 + nb - kb
    av = 1 + kv
    bv = 1 + nv - kv

    pv = np.linspace(0, 1, 10000)
    # p(pb < pv | pv)
    Ipv = scipy.special.betainc(ab, bb, pv)
    # p(pv)
    ppv = scipy.stats.beta.pdf(pv, av, bv)
    # p(pv < pv)
    p = 1-np.trapz(Ipv*ppv, pv)

    return p


def binom_confint(x):
    lower, upper = proportion_confint(
        x.sum(), len(x), alpha=0.05, method='jeffrey')
    return lower, upper


def binom_conf_lower(x):
    return binom_confint(x)[0]


def binom_conf_upper(x):
    return binom_confint(x)[1]


def evaluate_chance_model(responses):
    fields = ['order', 'feedback', 'ratio', 'pid', 'trial', 'stimulus']
    lh = responses[fields].copy()
    lh['lh'] = np.log(0.5)
    return lh


def evaluate_learning_model(responses, ipe, fb, fbratio):
    # extract trials and stimuli from the human data
    condinfo = responses[['trial', 'stimulus']].drop_duplicates()
    trials = np.array(condinfo.trial)
    stims = np.array(condinfo.stimulus)
    pids = responses.pid.unique()

    response_df = responses.pivot('pid', 'trial', 'fall? response')
    response_mat = response_df.as_matrix().astype('f8')

    fields = ['order', 'feedback', 'ratio', 'pid', 'trial', 'stimulus']
    lh = responses[fields].copy()

    try:
        float(fbratio)

    except ValueError:
        lh['lh'] = np.nan

    else:
        # convert the data frames to numpy arrays
        ipe_mat = make_ipe_mat(ipe, stims)
        fb_mat = make_fb_mat(fb, stims, 'fb', fbratio)

        # start with a uniform prior belief, regardless of model
        kappas = ipe.index.get_level_values('kappa').unique().astype('f8')
        prior = make_uniform_prior(kappas)

        lh_mat = mo.EvaluateObserverTrials(
            response_mat, fb_mat, ipe_mat, kappas,
            prior=prior, p_ignore=0.0, smooth=True)

        lh_df = pd.DataFrame(
            lh_mat.T,
            index=pd.MultiIndex.from_tuples(
                zip(trials, stims),
                names=['trial', 'stimulus']),
            columns=pd.Index(pids, name='pid')
        ).stack().reset_index().rename(columns={0: 'lh'})

        lh = pd.merge(lh, lh_df)
        lh.index = responses.index

    return lh


def evaluate_static_model(responses, ipe, fb):
    # extract trials and stimuli from the human data
    condinfo = responses[['trial', 'stimulus']].drop_duplicates()
    trials = np.array(condinfo.trial)
    stims = np.array(condinfo.stimulus)
    pids = responses.pid.unique()

    response_df = responses.pivot('pid', 'trial', 'fall? response')
    response_mat = response_df.as_matrix().astype('f8')

    fields = ['order', 'feedback', 'ratio', 'pid', 'trial', 'stimulus']
    lh = responses[fields].copy()

    # convert the data frames to numpy arrays
    ipe_mat = make_ipe_mat(ipe, stims)
    fb_mat = make_fb_mat(fb, stims, 'nfb', None)

    # start with a uniform prior belief, regardless of model
    kappas = ipe.index.get_level_values('kappa').unique().astype('f8')
    prior = make_uniform_prior(kappas)

    lh_mat = mo.EvaluateObserverTrials(
        response_mat, fb_mat, ipe_mat, kappas,
        prior=prior, p_ignore=0.0, smooth=True)

    lh_df = pd.DataFrame(
        lh_mat.T,
        index=pd.MultiIndex.from_tuples(
            zip(trials, stims),
            names=['trial', 'stimulus']),
        columns=pd.Index(pids, name='pid')
    ).stack().reset_index().rename(columns={0: 'lh'})

    lh = pd.merge(lh, lh_df)
    lh.index = responses.index

    return lh


def evaluate_model(responses, modelname, ipe, fb):

    if modelname == 'chance':
        lh = evaluate_chance_model(responses)

    elif modelname == 'learning':
        order, fbtype, ratio = responses.name
        lh = evaluate_learning_model(responses, ipe, fb, ratio)

    elif modelname == 'static':
        lh = evaluate_static_model(responses, ipe, fb)

    else:
        raise ValueError("unknown model '%s'" % modelname)

    lh['model'] = modelname
    return lh


def bootstrap(x, rso, nsamples=10000):
    arr = np.asarray(x)
    n, = arr.shape
    boot_idx = rso.randint(0, n, n*nsamples)
    boot_arr = arr[boot_idx].reshape((n, nsamples))
    boot_mean = boot_arr.mean(axis=0)
    stats = pd.Series(
        np.percentile(boot_mean, [15.8655254, 50, 84.1344746]),
        index=['lower', 'median', 'upper'],
        name=x.name)
    return stats


def bootstrap_test_gt(x, model1, model2, rso, nsamples=10000):
    groups = x.groupby(level='model')
    try:
        arr1 = np.asarray(groups.get_group(model1))
        arr2 = np.asarray(groups.get_group(model2))
    except KeyError:
        return np.nan

    assert arr1.shape == arr2.shape
    n, = arr1.shape
    boot_idx = rso.randint(0, n, n*nsamples)
    boot_arr1 = arr1[boot_idx].reshape((n, nsamples))
    boot_arr2 = arr2[boot_idx].reshape((n, nsamples))
    boot_mean1 = boot_arr1.mean(axis=0)
    boot_mean2 = boot_arr2.mean(axis=0)
    pct_gt = np.mean(boot_mean2 > boot_mean1)
    return pct_gt
