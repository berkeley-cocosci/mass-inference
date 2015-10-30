#!/usr/bin/env python

"""
Plots model vs human responses to "will it fall?" for a particular experiment
version and block.

"""

__depends__ = [
    "human_fall_responses.csv",
    "single_model_fall_responses.csv",
    "fall_response_corrs.csv"
]

import util
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot(dest, results_path, version, block, query):

    human = pd.read_csv(os.path.join(results_path, "human_fall_responses.csv"))
    human = human\
        .groupby(['version', 'block'])\
        .get_group((version, block))\
        .set_index(['kappa0', 'stimulus'])\
        .sortlevel()

    model = pd.read_csv(os.path.join(results_path, "single_model_fall_responses.csv"))
    model = model\
        .groupby(['query', 'block'])\
        .get_group((query, block))\
        .set_index(['kappa0', 'stimulus'])\
        .sortlevel()
    model = model.ix[human.index]

    fall_corrs = pd\
        .read_csv(os.path.join(results_path, "fall_response_corrs.csv"))\
        .set_index(['block', 'X', 'Y'])\
        .sortlevel()\
        .ix[(block, query, 'Human')]

    fig, ax = plt.subplots()

    # means
    mlerr = model['median'] - model['lower']
    muerr = model['upper'] - model['lower']
    hlerr = human['median'] - human['lower']
    huerr = human['upper'] - human['median']
    ax.plot([-0.03, 1.03], [-0.03, 1.03], '--', color=config['plots']['darkgrey'])
    ax.errorbar(
        model['median'], human['median'],
        xerr=[mlerr, muerr], yerr=[hlerr, huerr],
        color='k', linestyle='', marker='o')
    ax.set_xlabel(r"IPE observer model, $p(F_t|S_t)$")
    ax.set_ylabel(r"Normalized human judgments")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)

    pearson = r"$r={median:.2f}$, $95\%\ \mathrm{{CI}}\ [{lower:.2f},\ {upper:.2f}]$"
    corrstr = pearson.format(**fall_corrs)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text(
        xmax - (xmax - xmin) * 0.01, ymin + (ymax - ymin) * 0.035, corrstr,
        horizontalalignment='right', backgroundcolor='white')

    sns.despine()

    # adjust figure size
    fig.set_figwidth(3.25)
    fig.set_figheight(3.25)
    plt.draw()
    plt.tight_layout()

    # save to file
    for pth in dest:
        util.save(pth, close=False)


if __name__ == "__main__":
    config = util.load_config()
    parser = util.default_argparser(locals())
    parser.add_argument(
        '--version',
        default=config['analysis']['human_fall_version'],
        help='which version of the experiment to plot responses from')
    parser.add_argument(
        '--block',
        default='B',
        help='which block of the experiment to plot responses from')
    parser.add_argument(
        '--query',
        default=config['analysis']['query'],
        help='which ipe query to use')

    args = parser.parse_args()
    plot(args.to, args.results_path, args.version, args.block, args.query)
