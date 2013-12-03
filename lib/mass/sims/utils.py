from scenesim.objects.sso import SSO
import json
import numpy as np
from mass import SIM_PATH
from mass import SIM_SCRIPT_PATH as SCRIPT_PATH


def parse_address(address):
    host, port = address.split(":")
    return host, int(port)


def get_params(exp, tag):
    """Load the parameters from the simulation script."""
    sim_root = SIM_PATH.joinpath(exp, tag)
    script_root = SCRIPT_PATH.joinpath(exp, tag)
    script_file = script_root.joinpath("script.json")
    force_file = script_root.joinpath("force.npy")
    noise_file = script_root.joinpath("noise.npy")
    with script_file.open("r") as fid:
        script = json.load(fid)
    script["script_root"] = str(script_root)
    script["sim_root"] = str(sim_root)
    script["tasks_path"] = str(sim_root.joinpath("tasks.json"))
    script["forces"] = np.load(force_file)
    script["noises"] = np.load(noise_file)
    return script


def load_cpo(pth):
    """Load a cpo from disk."""
    with open(pth, "r") as fid:
        cpo = SSO.load_tree(fid)
    return cpo
