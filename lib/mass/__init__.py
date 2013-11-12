import panda3d.core as p3d
import scenesim
from path import path
import logging

# load panda configuration
ROOT_PATH = path(__path__[0]).joinpath("../../").abspath()
p3d.loadPrcFile(path.joinpath(ROOT_PATH, "Config.prc"))


def get_path(name):
    return ROOT_PATH.joinpath(
        p3d.ConfigVariableString(name, "").get_value())

CPO_PATH = get_path("cpo-path")
RENDER_PATH = get_path("render-path")
SIM_PATH = get_path("sim-path")
SIM_SCRIPT_PATH = get_path("sim-script-path")
RENDER_SCRIPT_PATH = get_path("render-script-path")
EXP_PATH = get_path("experiment-path")
DATA_PATH = get_path("data-path")
EGG_PATH = get_path("egg-path")
TEXTURE_PATH = get_path("texture-path")
BIN_PATH = get_path("bin-path")

p3d.getModelPath().appendDirectory(EGG_PATH)
p3d.getModelPath().appendDirectory(TEXTURE_PATH)

LOGLEVEL = p3d.ConfigVariableString("loglevel", "warn").get_value().upper()
FORMAT = '%(levelname)s -- %(processName)s/%(filename)s -- %(message)s'
logging.basicConfig(level=LOGLEVEL, format=FORMAT)
