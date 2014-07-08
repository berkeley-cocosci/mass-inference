from ConfigParser import SafeConfigParser
from path import path
from termcolor import colored
import pandas as pd

from . import __all__
from . import *
from . import util

root = path("..")

config = SafeConfigParser()
config.read(root.joinpath("config.ini"))
model_version = config.get("analysis", "model_version")
human_version = config.get("analysis", "human_version")
seed = config.getint("analysis", "seed")

data_path = root.joinpath(config.get("analysis", "data_path"))
results_path = root.joinpath(config.get("analysis", "results_path"))

exp_all, human = util.load_human(human_version, data_path)
data = util.load_all(model_version, data_path, human=human)

for name in __all__:
    func = locals()[name]
    print colored("Executing '%s'" % name, 'blue')
    pth = func.run(data, results_path, seed)
    print pth
    if pth.ext == '.csv':
        df = pd.read_csv(pth)
        print df
