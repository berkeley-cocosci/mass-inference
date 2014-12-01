#!/usr/bin/env python

import os
import glob
import json
from analyses.util import get_dependencies

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

with open(os.path.join(ROOT, "config.json"), "r") as fh:
    config = json.load(fh)

results_path = os.path.abspath(os.path.join(ROOT, config["paths"]["results"]))
data_path = os.path.abspath(os.path.join(ROOT, config["paths"]["data"]))

default_ext = {
    'analyses': '.csv',
    'plots': ['.pdf', '.png'],
    'latex_analyses': '.tex'
}

paths = {
    'analyses': os.path.join(ROOT, config["paths"]["results"]),
    'plots': os.path.join(ROOT, config["paths"]["figures"]),
    'latex_analyses': os.path.join(ROOT, config["paths"]["latex"])
}

dependencies = {}
for dirname in ['analyses', 'latex_analyses', 'plots']:
    files = glob.glob("{}/*.py".format(dirname))

    for filename in files:
        modname = os.path.splitext(os.path.basename(filename))[0]
        pkg = __import__(dirname, globals(), locals(), [modname], 0)
        mod = getattr(pkg, modname)

        if not hasattr(mod, '__depends__'):
            continue

        sources = get_dependencies(
            getattr(mod, '__depends__'),
            config, 
            results_path=results_path,
            data_path=data_path)[0]

        ext = getattr(mod, '__ext__', default_ext[dirname])
        if not hasattr(ext, '__iter__'):
            ext = [ext]
        targets = [os.path.join(paths[dirname], modname + x) for x in ext]

        dependencies['{}/{}.py'.format(dirname, modname)] = {
            'targets': targets,
            'sources': sources
        }

print json.dumps(dependencies)
