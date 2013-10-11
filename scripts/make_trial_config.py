import dbtools
import json
import numpy as np
import pandas as pd
from path import path


render_dir = path("../resources/render/G/")
json_dir = path("../experiment/static/json/")

conditions = [str(x.name) for x in render_dir.listdir()]
conditions = [x for x in conditions if x != "shared"]

base_conds = sorted(set(["-".join(x.split("-")[:-1]) for x in conditions]))
base_conds = dict(enumerate(base_conds))

with open(json_dir.joinpath("conditions.json"), "w") as fh:
    json.dump(base_conds, fh, indent=2)

DBPATH = path("../resources/sso/metadata.db")
tbl = dbtools.Table(DBPATH, "stability")

rso = np.random.RandomState(0)


def get_meta(stim, kappa):
    meta = tbl.select(where=("stimulus=? and kappa=?", (stim, kappa)))
    assert len(meta) == 1
    return meta

for condition in conditions:
    render_configs = render_dir.joinpath(condition).glob("*.csv")

    fb, ratio, cb = condition.split("-")
    if cb == "cb0":
        cb = 0
    elif cb == "cb1":
        cb = 1
    else:
        raise ValueError("invalid counterbalance: %s" % cb)

    config = {}
    for conf_path in render_configs:
        seed = abs(hash(conf_path.name))
        print conf_path, seed

        conf = pd.DataFrame.from_csv(conf_path).reset_index()
        conf.stimulus = conf.stimulus.map(lambda x: path(x).namebase)

        conf['feedback'] = fb
        conf['ratio'] = ratio
        conf['counterbalance'] = cb
        conf['fall? query'] = True
        conf['mass? query'] = True
        conf['color0'] = 'yellow'
        conf['color1'] = 'red'

        is_orig = conf.stimtype == "original"
        conf["mass? query"][is_orig] = False
        conf["color0"][is_orig] = None
        conf["color1"][is_orig] = None

        meta = pd.concat(map(get_meta, conf.stimulus, conf.kappa))
        meta['stable'] = meta['stable'].astype('bool')
        meta = meta.drop('dataset', axis=1)

        conf = pd.merge(conf, meta, on=['stimulus', 'kappa'])
        conf = (conf
                .drop(
                    ["full_render", "stimtype", "kappa", "flip_colors"],
                    axis=1)
                .set_index('stimulus')
                .sort()
                .reset_index())

        idx = np.array(conf.index)
        rso.seed(seed)
        rso.shuffle(idx)

        shuffled = conf.reindex(idx).reset_index(drop=True)
        shuffled.index.name = "trial"
        shuffled = shuffled.reset_index()

        dicts = shuffled.T.to_dict()
        trials = [dicts[i] for i in shuffled.index]

        if len(trials) == 1:
            trials = trials[0]

        config[conf_path.namebase] = trials

    with open(json_dir.joinpath("%s.json" % condition), "w") as fh:
        json.dump(config, fh, indent=2)
