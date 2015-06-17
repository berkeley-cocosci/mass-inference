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
    "model_log_lh_ratios.csv"
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
    if human is not None:
        timeseries(ax, human, colors['human'], ls=lines['human'])
    timeseries(ax, model.ix['learning'], colors['learning'], ls=lines['learning'])
    timeseries(ax, model.ix['static'], colors['static'], ls=lines['static'])


def add_llhr(ax, llhr):
    label = "LLR (learning v. static) = {:.2f}".format(llhr)
    l, h = ax.get_xlim()
    ax.text((h + l) / 2.0, 0.5125, label, horizontalalignment='center', fontsize=9)


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
        loc='lower center',
        fontsize=9,
        frameon=False)


def plot(dest, results_path, counterfactual, likelihood):

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
        .groupby(['likelihood', 'counterfactual', 'kappa0'])\
        .get_group((likelihood, counterfactual, 'all'))\
        .groupby(['version', 'num_mass_trials', 'trial', 'model', 'fitted'])\
        .filter(filter_trials)\
        .set_index(['version', 'fitted', 'model'])\
        .sortlevel()

    # load in the log likelihood ratios
    llhr = pd\
        .read_csv(os.path.join(results_path, 'model_log_lh_ratios.csv'))\
        .groupby(['likelihood', 'counterfactual', 'fitted'])\
        .get_group((likelihood, counterfactual, True))\
        .groupby(['version', 'num_mass_trials'])\
        .filter(filter_trials)\
        .set_index(['version'])['llhr']

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

    fig, axes = plt.subplots(2, 2, sharey=True)

    # top left subplot: ideal model for experiment 1
    plot_all(axes[0, 0], None, model.ix[('H', False)], lines, colors)
    axes[0, 0].set_title('(a) Ideal observer model')
    axes[0, 0].set_xlabel('Trial')
    axes[0, 0].set_ylabel('Pr(correct ratio chosen)')
    axes[0, 0].set_xticks([1, 5, 10, 15, 20])
    axes[0, 0].set_xlim([1, 20])
    axes[0, 0].set_ylim(0.48, 1.02)

    # top right subplot: experiment 1a
    plot_all(axes[0, 1], human.ix['H'], model.ix[('H', True)], lines, colors)
    axes[0, 1].set_title('(b) Experiment 1')
    axes[0, 1].set_xlabel('Trial')
    axes[0, 1].set_xticks([1, 5, 10, 15, 20])
    axes[0, 1].set_xlim([1, 20])
    add_llhr(axes[0, 1], llhr.ix['H'])

    # bottom left subplot: experiment 1b
    plot_all(axes[1, 0], human.ix['G'], model.ix[('G', True)], lines, colors)
    axes[1, 0].set_title('(c) Experiment 2')
    axes[1, 0].set_xlabel('Trial')
    axes[1, 0].set_ylabel('Pr(correct ratio chosen)')
    axes[1, 0].set_xticks([1, 2, 3, 4, 6, 9, 14, 20])
    axes[1, 0].set_xlim([1, 20])
    add_llhr(axes[1, 0], llhr.ix['G'])

    # bottom right subplot: experiment 2 (between subjects)
    plot_all(axes[1, 1], human.ix['I'], model.ix[('I', True)], lines, colors)
    axes[1, 1].set_title('(d) Experiment 3')
    axes[1, 1].set_xlabel('Trial')
    axes[1, 1].set_xticks([1, 2, 3, 5, 10])
    axes[1, 1].set_xlim([1, 10])
    add_llhr(axes[1, 1], llhr.ix['I'])

    # clear top and right axis lines
    sns.despine()

    # make the legend
    make_legend(axes[0, 0], lines, colors)

    # set figure size
    fig.set_figwidth(6)
    fig.set_figheight(5)
    plt.draw()
    plt.tight_layout()

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
        '--likelihood',
        default=config['analysis']['likelihood'],
        help='which version of the likelihood to plot')
    args = parser.parse_args()
    plot(args.to, args.results_path, args.counterfactual, args.likelihood)
