import dbtools
import json
import numpy as np
import pandas as pd
from path import path

conditions = {
    "0": {
        "pretest": "shared",
        "experimentA": "nfb-0.1",
        "experimentB": "vfb-0.1",
        "posttest": "shared",
    },

    "1": {
        "pretest": "shared",
        "experimentA": "nfb-0.1",
        "experimentB": "vfb-10",
        "posttest": "shared",
    },

    "2": {
        "pretest": "shared",
        "experimentA": "nfb-10",
        "experimentB": "vfb-0.1",
        "posttest": "shared",
    },

    "3": {
        "pretest": "shared",
        "experimentA": "nfb-10",
        "experimentB": "vfb-10",
        "posttest": "shared",
    },
}

render_dir = path("../resources/render/G/")
json_dir = path("../experiment/static/json/")

with open(json_dir.joinpath("conditions.json"), "w") as fh:
    json.dump(conditions, fh, indent=2, allow_nan=False)

DBPATH = path("../resources/sso/metadata.db")
tbl = dbtools.Table(DBPATH, "stability")

rso = np.random.RandomState(0)


def get_meta(stim, kappa):
    meta = tbl.select(where=("stimulus=? and kappa=?", (stim, kappa)))
    assert len(meta) == 1
    return meta


for condition, maps in conditions.iteritems():
    config = {}
    for phase, pth in maps.iteritems():
        for cb in [0, 1]:

            if phase in ("pretest", "posttest"):
                render_config = render_dir.joinpath(
                    pth, "%s.csv" % phase)

                fb = "vfb"
                ratio = 1
                ask_fall = True
                ask_mass = False

            else:
                render_config = render_dir.joinpath(
                    "%s-cb%d" % (pth, cb), "%s.csv" % phase)

                fb, ratio = pth.split("-")
                if phase == "experimentA":
                    ask_fall = True
                    ask_mass = False
                elif phase == "experimentB":
                    ask_fall = False
                    ask_mass = True

            seed = abs(hash(phase))
            print condition, phase, seed

            conf = pd.DataFrame.from_csv(render_config).reset_index()
            conf.stimulus = conf.stimulus.map(lambda x: path(x).namebase)

            conf['feedback'] = fb
            conf['ratio'] = ratio
            conf['counterbalance'] = cb
            conf['fall? query'] = ask_fall
            conf['mass? query'] = ask_mass

            meta = pd.concat(map(get_meta, conf.stimulus, conf.kappa))
            meta['stable'] = meta['stable'].astype('bool')
            meta = meta.drop('dataset', axis=1)

            conf = pd.merge(conf, meta, on=['stimulus', 'kappa'])
            conf = (conf
                    .drop(
                        ["full_render", "stimtype", "flip_colors"],
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

            for trial in trials:
                for key, val in trial.iteritems():
                    if isinstance(val, float) and np.isnan(val):
                        if key in ("color0", "color1"):
                            trial[key] = None

            if len(trials) == 1:
                trials = trials[0]

            config[phase] = trials

    with open(json_dir.joinpath("%s.json" % condition), "w") as fh:
        json.dump(config, fh, indent=2, allow_nan=False)
