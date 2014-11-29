#!/usr/bin/env python

"""
Create a csv file containing the trial order for each participant in the
experiment. The resulting file will have the following columns:

    version (string)
        experiment version
    kappa0 (float)
        true log mass ratio
    pid (string)
        the unique participant id
    mode (string)
        the experiment phase
    trial (int)
        the trial number
    stimulus (string)
        the stimulus name

"""

import util

def run(dest):
    human = util.load_human()
    order = human['all']\
        .set_index(['version', 'kappa0', 'pid', 'mode', 'trial'])[['stimulus']]\
        .sortlevel()

    order.to_csv(dest)

if __name__ == "__main__":
    parser = util.default_argparser(__doc__)
    args = parser.parse_args()
    run(args.dest)
