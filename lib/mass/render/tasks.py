import json
from path import path
import pandas as pd


def load_tasks(script_paths, force=False):
    scripts = []
    for script_path in script_paths:
        with open(script_path, "r") as fh:
            script = json.load(fh)
        df = pd.DataFrame.from_dict(script)
        df['script_path'] = str(script_path)
        df.index.name = "script_index"
        scripts.append(df)
    script = pd.concat(scripts).reset_index()
    if force:
        tasks = script.to_dict('list')
    else:
        tasks = script[~script['finished']].to_dict('list')
    return tasks


def mark_finished(script_path, i):
    with open(path(script_path), "r") as fh:
        script = json.load(fh)
    script['finished'][i] = True
    with open(path(script_path), "w") as fh:
        json.dump(script, fh)
