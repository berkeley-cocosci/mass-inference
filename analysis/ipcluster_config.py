c = get_config()

import os

# SSH-friendly name for the server and how many engines to start on it
SERVER = 'lovelace'
ENGINES = 8

# Paths to resources on the server
VIRTUALENV = '/home/jhamrick/.virtualenvs/mass-inference'
PROFILEDIR = '/home/jhamrick/.ipython/profile_default'
WORKDIR = '/home/jhamrick/project/mass-inference'

c.IPClusterStart.engine_launcher_class = 'SSH'
c.LocalControllerLauncher.controller_args = ["--ip='*'"]
c.SSHEngineSetLauncher.engines = {SERVER: ENGINES}
c.SSHEngineSetLauncher.engine_cmd = [os.path.join(VIRTUALENV, 'bin/python'), '-m', 'IPython.parallel.engine']
c.SSHEngineSetLauncher.engine_args = [
    '--profile-dir={}'.format(PROFILEDIR),
    '--work-dir={}'.format(os.path.join(WORKDIR, 'analysis/analyses'))
]
