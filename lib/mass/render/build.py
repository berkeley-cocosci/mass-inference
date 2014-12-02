"""Generate experiment render scripts."""

# Built-in
import json
import logging
# External
import numpy as np
import pandas as pd
# Local
from mass import CPO_PATH, RENDER_PATH
from mass import RENDER_SCRIPT_PATH as SCRIPT_PATH
from mass.stimuli import get_style

logger = logging.getLogger("mass.render")


def gen_angles(n, rso):
    angles = rso.randint(0, 360, n)
    return angles


def build(exp, condition, tag, force, cpo_path, seed, **params):

    # Path where we will save the rendered images/videos
    render_root = RENDER_PATH.joinpath(exp, condition)

    # Path where we will save the render script
    script_root = SCRIPT_PATH.joinpath(exp, condition)
    script_file = script_root.joinpath(tag + ".json")

    if script_file.exists() and not force:
        logging.debug("Script '%s' already exists", script_file.relpath())
        return

    # Location of the stimuli
    stims = sorted(CPO_PATH.joinpath(cpo_path).abspath().listdir())
    tasks = [x.namebase for x in stims]

    # Random number generator
    rso = np.random.RandomState(seed)

    # Start building the script!
    df = pd.DataFrame({
        'stimulus': stims,
        'task': tasks
    }).set_index('task')

    df['stimtype'] = get_style(cpo_path)
    df['camera_start'] = gen_angles(len(stims), rso)
    df['render_root'] = render_root
    df['finished'] = False

    for key in params:
        df[key] = params[key]

    # Convert it to a dictionary, so we can dump it to JSON
    script = df.to_dict('list')

    # Numpy bool types are not JSON-serializable, grumble
    script['feedback'] = map(bool, script['feedback'])
    script['finished'] = map(bool, script['finished'])
    script['full_render'] = map(bool, script['full_render'])
    script['occlude'] = map(bool, script['occlude'])

    # Save it
    if not script_root.exists():
        logger.debug("Creating directory %s", script_root)
        script_root.makedirs_p()

    with open(script_file, "w") as fh:
        json.dump(script, fh)
    logger.info("Saved script to %s", script_file.relpath())
