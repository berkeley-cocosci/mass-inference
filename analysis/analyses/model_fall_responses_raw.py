#!/usr/bin/env python

"""
Computes the average model queries for "will it fall?". The results are saved
into a HDF5 database, with the following structure:

    <query>/params_<n>

where <query> is the name of the query (for example, 'percent_fell') and
params_<n> (e.g. 'params_0') is the particular combination of sigma/phi
parameters. Additionally, there is a <query>/param_ref array that gives a
mapping between the actual parameter values and their identifiers.

Each table in the database has the following columns:

    query (string)
        the model query
    block (string)
        the experiment phase
    kappa0 (float)
        the true log mass ratio
    stimulus (string)
        the name of the stimulus
    sample (int)
        sample number
    response (float in [0, 1])
        query response

"""

__depends__ = ["ipe_A", "ipe_B"]
__ext__ = '.h5'

import util
import pandas as pd
import model_fall_responses_queries as queries

def model_fall_responses(queryname, data):
    result = data\
        .groupby(['block', 'stimulus'])\
        .apply(getattr(queries, queryname))\
        .stack()\
        .to_frame('response')\
        .reset_index()\
        .rename(columns=dict(kappa='kappa0'))
    result['query'] = queryname
    return result


def run(dest, data_path):
    # load the raw ipe data
    ipe = util.load_ipe(data_path)
    groups = ipe.groupby(['sigma', 'phi'])

    # open up the store for saving
    store = pd.HDFStore(dest, mode='w')

    # process the data
    all_params = {}
    for i, params in enumerate(groups.groups.keys()):
        all_params['params_{}'.format(i)] = params

        for query in queries.__all__:
            key = "/{}/params_{}".format(query, i)
            print(key)

            data = groups.get_group(params)
            result = model_fall_responses(query, data)
            store.append(key, result)

    # save the parameters into the database
    all_params = pd.DataFrame(all_params, index=['sigma', 'phi']).T
    for query in queries.__all__:
        key = "/{}/param_ref".format(query)
        store.append(key, all_params)

    store.close()


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    try:
        run(args.to, args.data_path)
    except:
        if os.path.exists(args.to):
            os.remove(args.to)

