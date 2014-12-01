#!/usr/bin/env python

"""
Computes the number of stimuli for which people were not significantly above
chance in judging which mass was heavier. Produces a csv file with the following
columns:

    version (string)
        the experiment version
    kappa0 (float)
        the true log mass ratio
    stimulus (string)
        the stimulus name
    0.00125 (boolean)
        whether p(p(correct)) < 0.00125

"""

__depends__ = ["human"]

import util

def run(dest, data_path):
    human = util.load_human(data_path)
    results = []

    def num_chance(df):
        groups = df.groupby(['kappa0', 'stimulus'])['mass? correct']
        alpha = 0.05 / len(groups.groups)
        results = groups\
            .apply(lambda x: util.beta(x, 1, [alpha]))\
            .unstack(-1) <= 0.5
        return results

    results = human['C']\
        .dropna(axis=0, subset=['mass? response'])\
        .groupby('version')\
        .apply(num_chance)

    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.data_path)
