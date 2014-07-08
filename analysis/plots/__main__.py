from ConfigParser import SafeConfigParser
from path import path
from termcolor import colored
import traceback

from . import __all__
from . import *
from . import util

root = path("../")

config = SafeConfigParser()
config.load(root.joinpath("config.ini"))

results_path = root.joinpath(config.get("analysis", "results_path"))
fig_path = root.joinpath(config.get("analysis", "figure_path"))

for name in __all__:
    func = locals()[name]
    print colored("Executing '%s'" % name, 'blue')
    try:
        print func.plot(results_path, fig_path)
    except:
        print colored(traceback.format_exc(limit=3), "red")
