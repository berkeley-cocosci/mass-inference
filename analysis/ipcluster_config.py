c = get_config()

c.IPClusterStart.engine_launcher_class = 'SSH'
c.LocalControllerLauncher.controller_args = ["--ip='*'"]
c.SSHEngineSetLauncher.engines = {
    'lovelace': 8
}
c.SSHEngineSetLauncher.engine_cmd = ['/home/jhamrick/.virtualenvs/mass-inference/bin/python', '-m', 'IPython.parallel.engine']
c.SSHEngineSetLauncher.engine_args = ['--profile-dir=/home/jhamrick/.ipython/profile_default', '--work-dir=/home/jhamrick/project/mass-inference/analysis/analyses']
