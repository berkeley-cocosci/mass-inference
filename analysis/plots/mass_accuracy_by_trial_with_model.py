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


def plot_models(ax, x, lines, colors):
    for key in lines:
        timeseries(ax, x.ix[key], color=colors[key], ls=lines[key])


def add_llhr(ax, llhr):
    label = "{} LLR = {:.2f}"
    text = "{}\n{}".format(
        label.format("Empirical", llhr.ix["empirical"]),
        label.format("IPE", llhr.ix["ipe"]))
    l, h = ax.get_xlim()
    ax.text(h, 0.5125, text, horizontalalignment='right', fontsize=9)

def filter_trials(df):
    return (df['version'] != 'I') | (df['num_mass_trials'] == -1) 


def make_legend(ax, lines, colors):
    # TODO: figure out how to add the legend

    # label = "{} model,\n".format(model.capitalize())
    # if lh == 'ipe':
    #     label = "{}IPE likelihod".format(label)
    # elif lh == "empirical":
    #     label = "{}Emp. likelihood".format(label)

    # labels, lines = zip(*sorted(lines.items()))

    # ax.legend(
    #     lines, labels,
    #     bbox_to_anchor=[1.05, 0.5],
    #     loc='center left',
    #     fontsize=9,
    #     frameon=False)

    pass


def plot(dest, results_path):

    # load in the human responses
    human = pd\
        .read_csv(os.path.join(results_path, 'human_mass_accuracy_by_trial.csv'))\
        .groupby('kappa0')\
        .get_group('all')\
        .groupby(['version', 'num_mass_trials'])\
        .filter(filter_trials)\
        .set_index('version')

    # load in the model responses
    model = pd\
        .read_csv(os.path.join(results_path, 'model_mass_accuracy_by_trial.csv'))\
        .groupby(['counterfactual', 'fitted', 'kappa0'])\
        .get_group((True, True, 'all'))\
        .groupby(['version', 'num_mass_trials'])\
        .filter(filter_trials)\
        .set_index(['version', 'likelihood', 'model'])\
        .sortlevel()

    # load in the log likelihood ratios
    llhr = pd\
        .read_csv(os.path.join(results_path, 'model_log_lh_ratios.csv'))\
        .groupby(['counterfactual', 'fitted'])\
        .get_group((True, True))\
        .groupby(['version', 'num_mass_trials'])\
        .filter(filter_trials)\
        .set_index(['version', 'likelihood'])['llhr']

    # colors and line styles
    plot_config = util.load_config()["plots"]
    darkgrey = plot_config["darkgrey"]
    palette = sns.color_palette("Dark2")
    colors = {
        ('ipe', 'static'): palette[0],
        ('ipe', 'learning'): palette[0],
        ('empirical', 'static'): palette[1],
        ('empirical', 'learning'): palette[1]
    }
    lines = {
        ('ipe', 'static'): '--',
        ('ipe', 'learning'): '-',
        ('empirical', 'static'): '--',
        ('empirical', 'learning'): '-'
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    # left subplot: experiment 1
    timeseries(ax1, human.ix['H'], darkgrey)
    plot_models(ax1, model.ix['H'], lines, colors)
    ax1.set_title('Experiment 1')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Fraction correct')
    ax1.set_xticks([1, 5, 10, 15, 20])
    ax1.set_xlim([1, 20])
    ax1.set_ylim(0.5, 1)
    add_llhr(ax1, llhr.ix['H'])

    # middle subplot: experiment 2
    timeseries(ax2, human.ix['G'], darkgrey)
    plot_models(ax2, model.ix['G'], lines, colors)
    ax2.set_title('Experiment 2')
    ax2.set_xlabel('Trial')
    ax2.set_xticks([1, 2, 3, 4, 6, 9, 14, 20])
    ax2.set_xlim([1, 20])
    add_llhr(ax2, llhr.ix['G'])

    # right subplot: experiment 3 (between subjects)
    timeseries(ax3, human.ix['I'], darkgrey)
    plot_models(ax3, model.ix['I'], lines, colors)
    ax3.set_title('Experiment 3')
    ax3.set_xlabel('Trial')
    ax3.set_xticks([1, 2, 3, 5, 10])
    ax3.set_xlim([1, 10])
    add_llhr(ax3, llhr.ix['I'])

    # clear top and right axis lines
    sns.despine()

    # make the legend
    make_legend(ax3, lines, colors)

    # set figure size
    fig.set_figwidth(9)
    fig.set_figheight(3)
    plt.draw()
    plt.tight_layout()
    plt.subplots_adjust(right=0.825)

    # save
    for pth in dest:
        util.save(pth, close=False)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    plot(args.to, args.results_path)
