#!/usr/bin/env python

"""
Plots accuracy on "which is heavier?" as a function of trial, both for the model
and participants, in all three experiments (for experiment 3, it is
between-subjects). There are four versions of the model plotted: both ipe and
empirical likelihoods, and both static and learning models.
"""

__depends__ = [
    "human_mass_accuracy_by_trial.csv", 
    "model_mass_accuracy_by_trial.csv",
    "model_log_lh_ratios.csv",
    "bayes_factors.csv"
]

import util
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import seaborn as sns


def timeseries(ax, x, color, ls='-', label=None):
    ax.fill_between(
        x['trial'], x['lower'], x['upper'],
        color=color, alpha=0.3)
    ax.plot(
        x['trial'], x['median'], 
        lw=2, ls=ls, color=color, label=label)

    ax.set_xlim(x['trial'].min(), x['trial'].max())


def plot_all(ax, human, model, lines, colors):
    timeseries(ax, human, colors['human'], ls=lines['human'])
    timeseries(ax, model.ix['learning'], colors['learning'], ls=lines['learning'])
    timeseries(ax, model.ix['static'], colors['static'], ls=lines['static'])


def add_llhr_and_factor(ax, llhr, factor):
    label = (
        "LLR (learning v. static) = {:.2f}\n"
        "Bayes Factor = {:.2f}"
    ).format(llhr, factor)
    l, h = ax.get_xlim()
    ax.text((h + l) / 2.0, 0.5125, label, horizontalalignment='center')


def filter_trials(df):
    return (df['version'] != 'I') | (df['num_mass_trials'] == -1)


def make_legend(ax, lines, colors):
    handles = []

    for model in sorted(colors.keys()):
        if model == 'human':
            label = 'Human'
        else:
            label = "{} model".format(model.capitalize())

        handles.append(mlines.Line2D(
            [], [],
            color=colors[model], 
            linestyle=lines[model],
            label=label))

    ax.legend(
        handles=handles,
        bbox_to_anchor=[1.05, 0.5],
        loc='center left',
        frameon=False)


def plot(dest, results_path, counterfactual, fitted, likelihood):
    if likelihood == 'ipe':
        likelihood = 'ipe_' + util.get_query()

    # load in the human responses
    human = pd\
        .read_csv(os.path.join(results_path, 'human_mass_accuracy_by_trial.csv'))\
        .groupby('kappa0')\
        .get_group('all')\
        .groupby(['version', 'num_mass_trials', 'trial'])\
        .filter(filter_trials)\
        .set_index('version')

    # load in the model responses
    model = pd\
        .read_csv(os.path.join(results_path, 'model_mass_accuracy_by_trial.csv'))\
        .groupby(['likelihood', 'counterfactual', 'fitted', 'kappa0'])\
        .get_group((likelihood, counterfactual, fitted, 'all'))\
        .groupby(['version', 'num_mass_trials', 'trial', 'model'])\
        .filter(filter_trials)\
        .set_index(['version', 'model'])\
        .sortlevel()

    # load in the log likelihood ratios
    llhr = pd\
        .read_csv(os.path.join(results_path, 'model_log_lh_ratios.csv'))\
        .groupby(['likelihood', 'counterfactual', 'fitted'])\
        .get_group((likelihood, counterfactual, fitted))\
        .groupby(['version', 'num_mass_trials'])\
        .filter(filter_trials)\
        .set_index(['version'])['llhr']

    # load in the bayes factors
    factors = pd\
        .read_csv(os.path.join(results_path, 'bayes_factors.csv'))\
        .groupby(['likelihood', 'counterfactual'])\
        .get_group((likelihood, counterfactual))\
        .groupby(['version', 'num_mass_trials'])\
        .filter(filter_trials)\
        .set_index(['version'])['logK']

    # colors and line styles
    plot_config = util.load_config()["plots"]
    darkgrey = plot_config["darkgrey"]
    lines = {
        'human': '-',
        'static': ':',
        'learning': '--'
    }
    colors = {
        'human': darkgrey,
        'static': darkgrey,
        'learning': darkgrey
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    # left subplot: experiment 1
    plot_all(ax1, human.ix['H'], model.ix['H'], lines, colors)
    ax1.set_title('Experiment 1')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Fraction correct')
    ax1.set_xticks([1, 5, 10, 15, 20])
    ax1.set_xlim([1, 20])
    ax1.set_ylim(0.5, 1)
    add_llhr_and_factor(ax1, llhr.ix['H'], factors.ix['H'])

    # middle subplot: experiment 2
    plot_all(ax2, human.ix['G'], model.ix['G'], lines, colors)
    ax2.set_title('Experiment 2')
    ax2.set_xlabel('Trial')
    ax2.set_xticks([1, 2, 3, 4, 6, 9, 14, 20])
    ax2.set_xlim([1, 20])
    add_llhr_and_factor(ax2, llhr.ix['G'], factors.ix['G'])

    # right subplot: experiment 3 (between subjects)
    plot_all(ax3, human.ix['I'], model.ix['I'], lines, colors)
    ax3.set_title('Experiment 3')
    ax3.set_xlabel('Trial')
    ax3.set_xticks([1, 2, 3, 5, 10])
    ax3.set_xlim([1, 10])
    add_llhr_and_factor(ax3, llhr.ix['I'], factors.ix['I'])

    # clear top and right axis lines
    sns.despine()

    # make the legend
    make_legend(ax3, lines, colors)

    # set figure size
    fig.set_figwidth(6.5)
    fig.set_figheight(2.15)
    plt.draw()
    plt.tight_layout()
    plt.subplots_adjust(right=0.825)

    # save
    for pth in dest:
        util.save(pth, close=False)


if __name__ == "__main__":
    config = util.load_config()
    parser = util.default_argparser(locals())
    if config['analysis']['counterfactual']:
        parser.add_argument(
            '--no-counterfactual',
            action='store_false',
            dest='counterfactual',
            default=True,
            help="don't plot the counterfactual likelihoods")
    else:
        parser.add_argument(
            '--counterfactual',
            action='store_true',
            dest='counterfactual',
            default=False,
            help='plot the counterfactual likelihoods')
    parser.add_argument(
        '--not-fitted',
        action='store_false',
        dest='fitted',
        default=True,
        help='whether to plot the fitted models')
    parser.add_argument(
        '--likelihood',
        default=config['analysis']['likelihood'],
        help='which version of the likelihood to plot')

    args = parser.parse_args()
    plot(args.to, args.results_path, args.counterfactual, args.fitted, args.likelihood)
