#!/usr/bin/env python

"""
Create a csv file containing the trial order for each participant in the
experiment. The resulting file will have the following columns:

    mode (string)
        the experiment phase
    trial (int)
        the trial number
    pid (string)
        the unique participant id
    stimulus (string)
        the stimulus name

"""

import util


def run(results_path):
    human = util.load_human()
    order = human['all']\
        .set_index(['mode', 'trial', 'pid'])['stimulus']

    order.to_csv(results_path)


if __name__ == "__main__":
    parser = util.default_argparser(__doc__, add_seed=False)
    args = parser.parse_args()
    run(args.results_path)
