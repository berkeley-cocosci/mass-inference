from path import path
from termcolor import colored
import pandas as pd

from . import __all__
from . import *
from . import util

root = path("../")
version = 'G'
data_path = root.joinpath('data')
data = util.load_all(version, data_path)
results_path = root.joinpath('results')
seed = 923012

for name in __all__:
    func = locals()[name]
    print colored("Executing '%s'" % name, 'blue')
    pth = func.run(data, results_path, seed)
    print pth
    if pth.ext == '.csv':
        df = pd.read_csv(pth)
        print df
