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

__depends__ = ["human"]

import util


def run(dest, data_path):
    human = util.load_human(data_path)

    # compute how many participants we have for each condition
    counts = human['all']\
        .groupby(['version', 'condition', 'counterbalance'])['pid']\
        .apply(lambda x: len(x.unique()))\
        .reset_index()
    counts.columns = [
        'version', 'condition', 'counterbalance', 'num_participants']
    counts = counts.set_index(['version', 'condition', 'counterbalance'])

    counts.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.dest, args.data_path)
