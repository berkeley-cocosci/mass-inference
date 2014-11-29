#!/usr/bin/env python

"""
Computes the number of participants in each condition of each experiment.
Outputs a csv with the following columns:

    version (string)
        the version of the experiment
    condition (int)
        the number of the condition
    counterbalance (bool)
        boolean indicating counterbalancing
    num_participants (int)
        how many participants were in the respective 
        version/condition/counterbalance

"""

import util


def run(results_path):
    human = util.load_human()

    # compute how many participants we have for each condition
    counts = human['all']\
        .groupby(['version', 'condition', 'counterbalance'])['pid']\
        .apply(lambda x: len(x.unique()))\
        .reset_index()
    counts.columns = [
        'version', 'condition', 'counterbalance', 'num_participants']
    counts = counts.set_index(['version', 'condition', 'counterbalance'])

    counts.to_csv(results_path)


if __name__ == "__main__":
    parser = util.default_argparser(__doc__, add_seed=False)
    args = parser.parse_args()
    run(args.results_path)
