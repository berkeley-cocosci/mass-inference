#!/usr/bin/env python

"""
Computes average participant responses to "which is heavier?". Produces a csv
file with the following columns:

    version (string)
        experiment version
    kappa0 (float)
        true log mass ratio
    stimulus (string)
        stimulus name
    pid (string)
        participant id
    mass? response (int)
        the participants' judgment, either 0 or 1

"""

__depends__ = ["human"]

import util


def run(dest, data_path):
    human = util.load_human(data_path)['C']\
        .dropna(axis=0, subset=['mass? correct'])
    results = human.set_index(['version', 'kappa0', 'stimulus', 'pid'])[['mass? correct']]
    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.data_path)
