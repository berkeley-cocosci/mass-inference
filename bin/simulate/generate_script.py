#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mass.sims.build import build
from copy import deepcopy
import numpy as np

defaults = dict(
    # maximum number of conditions per simulation task
    max_chunk_size=250,

    # number of simulation samples
    num_samples=None,

    # random seed
    seed=2938,

    # path to the floor
    floor_path="floors/round-wooden-floor.cpo",

    # standard deviation of position noise
    sigmas=None,

    # force magnitude
    phis=None,

    # log10 mass ratios
    kappas=[
        -1.3, -1.2, -1.1, -1.0, -0.9, -0.8,
        -0.7, -0.6, -0.5, -0.4, -0.3, -0.2,
        -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3
    ],

    # physics parameters
    physics={
        'gravity': [0.0, 0.0, -9.81],
        'force_duration': 0.2
    },

    # general simulation/recording parameters
    simulation={
        # total length (in seconds) of the simulation
        'duration': 2.,

        # time in seconds for each physics step
        'step_size': 0.01,

        # time in seconds for each physics substep
        'substep_size': 1. / 1000,

        # how often (in steps) we record data
        'record_interval': 10,
    },
)


def make_options():
    options = []

    # stability original truth
    options.append({
        'exp': 'stability_original',
        'tag': 'truth',
        'cpo_path': "stability-original",
        'num_samples': 1,
        'sigmas': [0.0],
        'phis': [0.0],
        'kappas': [0.0]
    })

    # stability original ipe
    options.append({
        'exp': 'stability_original',
        'tag': 'ipe',
        'cpo_path': "stability-original",
        'num_samples': 100,
        'sigmas': [0.04],
        'phis': [0.2],
        'kappas': [0.0]
    })

    # stability sameheight truth
    options.append({
        'exp': 'stability_sameheight',
        'tag': 'truth',
        'cpo_path': "stability-sameheight",
        'num_samples': 1,
        'sigmas': [0.0],
        'phis': [0.0],
        'kappas': [0.0]
    })

    # stability sameheight ipe
    options.append({
        'exp': 'stability_sameheight',
        'tag': 'ipe',
        'cpo_path': "stability-sameheight",
        'num_samples': 100,
        'sigmas': [0.04],
        'phis': [0.2],
        'kappas': [0.0]
    })

    # mass all truth
    options.append({
        'exp': 'mass_all',
        'tag': 'truth',
        'cpo_path': "mass-all",
        'num_samples': 1,
        'sigmas': [0.0],
        'phis': [0.0],
        'kappas': [-1.0, -0.3, 0.0, 0.3, 1.0]
    })

    # mass all ipe
    options.append({
        'exp': 'mass_all',
        'tag': 'ipe',
        'cpo_path': "mass-all",
        'num_samples': 10,
        'sigmas': [0.04],
        'phis': [0.2],
        'kappas': [-1.0, -0.3, 0.0, 0.3, 1.0]
    })

    # mass learning truth
    options.append({
        'exp': 'mass_learning',
        'tag': 'truth',
        'cpo_path': "mass-learning",
        'num_samples': 1,
        'sigmas': [0.0],
        'phis': [0.0]
    })

    # mass learning ipe
    options.append({
        'exp': 'mass_learning',
        'tag': 'ipe',
        'cpo_path': "mass-learning",
        'num_samples': 100,
        'sigmas': [0.04],
        'phis': [0.2]
    })

    # mass prediction stability truth
    options.append({
        'exp': 'mass_prediction_stability',
        'tag': 'truth',
        'cpo_path': "mass-prediction-stability",
        'num_samples': 1,
        'sigmas': [0.0],
        'phis': [0.0]
    })

    # mass prediction stability ipe
    options.append({
        'exp': 'mass_prediction_stability',
        'tag': 'ipe',
        'cpo_path': "mass-prediction-stability",
        'num_samples': 100,
        'sigmas': [0.04],
        'phis': [0.2]
    })

    # mass prediction direction truth
    options.append({
        'exp': 'mass_prediction_direction',
        'tag': 'truth',
        'cpo_path': "mass-prediction-direction",
        'num_samples': 1,
        'sigmas': [0.0],
        'phis': [0.0]
    })

    # mass prediction direction ipe
    options.append({
        'exp': 'mass_prediction_direction',
        'tag': 'ipe',
        'cpo_path': "mass-prediction-direction",
        'num_samples': 100,
        'sigmas': [0.04],
        'phis': [0.2]
    })

    # G-a-truth
    options.append({
        'exp': 'mass_inference-G-a',
        'tag': 'truth',
        'cpo_path': "mass-inference-G-a",
        'num_samples': 1,
        'sigmas': [0.0],
        'phis': [0.0]
    })

    # G-a-ipe
    options.append({
        'exp': 'mass_inference-G-a',
        'tag': 'ipe',
        'cpo_path': "mass-inference-G-a",
        'num_samples': 100,
        'sigmas': [0.04],
        'phis': [0.2]
    })

    # G-b-truth
    options.append({
        'exp': 'mass_inference-G-b',
        'tag': 'truth',
        'cpo_path': "mass-inference-G-b",
        'num_samples': 1,
        'sigmas': [0.0],
        'phis': [0.0]
    })

    # G-b-ipe
    options.append({
        'exp': 'mass_inference-G-b',
        'tag': 'ipe',
        'cpo_path': "mass-inference-G-b",
        'num_samples': 100,
        'sigmas': [0.04],
        'phis': [0.2]
    })

    # I-a-truth
    options.append({
        'exp': 'mass_inference-I-a',
        'tag': 'truth',
        'cpo_path': "mass-inference-I-a",
        'num_samples': 1,
        'sigmas': [0.0],
        'phis': [0.0],
        'kappas': [-1.0, 0.0, 1.0]
    })

    # I-a-ipe
    options.append({
        'exp': 'mass_inference-I-a',
        'tag': 'ipe',
        'cpo_path': "mass-inference-I-a",
        'num_samples': 100,
        'sigmas': [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03,
                   0.035, 0.04, 0.045, 0.05],
        'phis': [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18,
                 0.21, 0.24, 0.27, 0.30],
        'kappas': [-1.0, 0.0, 1.0]
    })

    # I-b-truth
    options.append({
        'exp': 'mass_inference-I-b',
        'tag': 'truth',
        'cpo_path': "mass-inference-I-b",
        'num_samples': 1,
        'sigmas': [0.0],
        'phis': [0.0],
        'kappas': [-1.0, 0.0, 1.0]
    })

    # I-b-ipe
    options.append({
        'exp': 'mass_inference-I-b',
        'tag': 'ipe',
        'cpo_path': "mass-inference-I-b",
        'num_samples': 100,
        'sigmas': [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03,
                   0.035, 0.04, 0.045, 0.05],
        'phis': [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18,
                 0.21, 0.24, 0.27, 0.30],
        'kappas': [-1.0, 0.0, 1.0]
    })

    return options


def make_parser(exps, tags):
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-e", "--exp",
        required=True,
        choices=exps,
        help="Experiment version.")
    parser.add_argument(
        "-t", "--tag",
        required=True,
        choices=tags,
        help="Short name to identify this stimulus set.")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Force script to be generated.")

    return parser


if __name__ == "__main__":
    options = make_options()
    exps = sorted(set([o['exp'] for o in options]))
    tags = sorted(set([o['tag'] for o in options]))

    parser = make_parser(exps, tags)
    args = parser.parse_args()

    for opts in options:
        if args.exp == opts['exp'] and args.tag == opts['tag']:
            params = deepcopy(defaults)
            params.update(opts)
            params['exp'] = args.exp
            params['force'] = args.force
            build(**params)
            break
