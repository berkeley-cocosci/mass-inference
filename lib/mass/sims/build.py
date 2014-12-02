"""Generate IPE simulation scripts."""

# Built-in
import json
import logging
# External
import numpy as np
# Scenesim
from scenesim.objects.pso import PSO
from scenesim.objects.sso import SSO
# Local
from mass import CPO_PATH, SIM_PATH
from mass import SIM_SCRIPT_PATH as SCRIPT_PATH

logger = logging.getLogger("mass.sims")


def get_objects(cpo_path):
    """Get the names of all of an SSO's children"""
    sso = SSO.load_tree(cpo_path)
    children = sso.descendants(type_=PSO)
    objs = [x.getName() for x in children]
    return objs


def build_noises(sigmas, shape, rso):
    """Generate an array of position noise for each sigma.

    Parameters
    ----------
    sigmas : 1d array-like
        List of standard deviations of the noise
    shape : tuple of ints
        Shape to generating random noise in
    rso : np.random.RandomState
        Random number generator

    """
    # allocate the array
    noises = np.empty((len(sigmas),) + shape + (3,))
    # generate the values
    for i, sigma in enumerate(sigmas):
        if sigma == 0:
            noises[i] = 0
        else:
            noises[i] = rso.normal(0, sigma, shape + (3,))
    return noises


def build_forces(phis, shape, rso):
    """Generate an array of force noise for each phi. Angles are chosen
    evenly distributed around the circle.

    Parameters
    ----------
    phi : 1d array-like
        List of force magnitudes
    shape : tuple of ints
        Shape to generating random noise in
    rso : np.random.RandomState
        Random number generator

    """
    # create the datatype we'll be using
    dtype = np.dtype([
        ('dir', 'f8'),
        ('mag', 'f8')
    ])
    # allocate the array
    forces = np.empty((len(phis),) + shape, dtype=dtype)
    # generate the random directions
    forces['dir'] = rso.randint(0, 360, shape)
    forces['mag'] = np.array(phis)[[slice(None)] + [None] * len(shape)]
    return forces


def build_records(params):
    """Compute the list of step on which to record simulation data."""

    # load the various simulation parameters
    sim_duration = float(params['duration'])
    step_size = float(params['step_size'])
    substep_size = float(params['substep_size'])
    record_interval = float(params['record_interval'])

    # number of substeps per step
    n_substeps = int(np.floor(step_size / substep_size))
    # number of steps in the whole simulation
    n_steps = int(np.ceil(sim_duration / step_size))
    # number times we record during the whole simulation
    n_strides = int(np.ceil(n_steps / record_interval))

    # compute the steps, always making sure to include the last one
    strides = np.round(np.linspace(0, n_steps, n_strides + 1)).astype(int)
    record_steps = map(int, np.unique(np.hstack([strides, n_steps])))

    # also include the initial "pre-repel" step
    return ["pre-repel"] + record_steps, n_substeps


def build(exp, tag, force, **params):
    """Create a simulation script."""

    # Path where we will save the simulations
    sim_root = SIM_PATH.joinpath(exp, tag)

    # Path where we will save the simulation script/resources
    script_root = SCRIPT_PATH.joinpath(exp, tag)
    script_file = script_root.joinpath("script.json")
    noise_file = script_root.joinpath("noise.npy")
    force_file = script_root.joinpath("force.npy")

    # check to see if we would override existing data
    if not force and script_root.exists():
        logger.debug("Script %s already exists", script_root.relpath())
        return

    # remove existing files, if we're overwriting
    if script_root.exists():
        script_root.rmtree()

    # Locations of stimuli and the floor
    cpo_paths = sorted(CPO_PATH.joinpath(params['cpo_path']).listdir())
    floor_path = CPO_PATH.joinpath(params['floor_path'])

    # Names of all the objects we'll be simulating -- make sure
    # they're the same for all scenes, because we currently don't
    # support them being different
    objs = [get_objects(x) for x in cpo_paths]
    assert (np.array(objs)[[0]] == np.array(objs)).all()

    # Create a random number generator
    rso = np.random.RandomState(params['seed'])

    # Determine the shape we need for generating sigmas/phis
    n_stims = len(cpo_paths)
    n_samples = params['num_samples']
    n_objs = len(objs[0])

    # Generate arrays of perceptual and force noise
    noises = build_noises(
        params['sigmas'], (n_stims, n_samples, n_objs), rso)
    forces = build_forces(
        params['phis'], (n_stims, n_samples), rso)

    # The timesteps when we will actually be recording
    record_steps, n_substeps = build_records(params['simulation'])

    # Put it all together in a big dictionary...
    script = {}

    # Simulation version
    script['exp'] = exp
    script['tag'] = tag

    # Various paths -- but strip away the absolute parts, because we
    # might be running the simulations on another computer
    script['sim_root'] = str(sim_root.relpath(SIM_PATH))
    script['cpo_paths'] = [str(x.relpath(CPO_PATH)) for x in cpo_paths]
    script['floor_path'] = str(floor_path.relpath(CPO_PATH))
    script['noise_path'] = str(noise_file.relpath(script_root))
    script['force_path'] = str(force_file.relpath(script_root))

    # physics parameters and simulation parameters
    script['physics'] = params['physics']
    script['simulation'] = params['simulation']
    script['simulation']['n_substeps'] = n_substeps
    script['max_chunk_size'] = params['max_chunk_size']

    # the index names for the data we'll be saving out
    script['index_names'] = [
        "sigma",
        "phi",
        "kappa",
        "stimulus",
        "sample",
        "timestep",
        "object",
        "posquat"
    ]

    # the corresponding level names for that index
    script['index_levels'] = {
        'sigma': params['sigmas'],
        'phi': params['phis'],
        'kappa': params['kappas'],
        'stimulus': [str(x.name) for x in cpo_paths],
        'sample': range(n_samples),
        'object': objs[0],
        'timestep': record_steps,
        'posquat': ['x', 'y', 'z', 'q0', 'q1', 'q2', 'q3'],
    }

    # create the directory for our script and resources, and save them
    script_root.makedirs_p()

    with script_file.open("w") as fh:
        json.dump(script, fh, indent=2)
    np.save(noise_file, noises)
    np.save(force_file, forces)
    logger.info("Saved script to %s", script_root.relpath())

    return script
