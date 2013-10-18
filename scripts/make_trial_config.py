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
        "unstable_example": "shared",
        "stable_example": "shared",
        "mass_example": "nfb-0.1"
    },

    "1": {
        "pretest": "shared",
        "experimentA": "nfb-0.1",
        "experimentB": "vfb-10",
        "posttest": "shared",
        "unstable_example": "shared",
        "stable_example": "shared",
        "mass_example": "nfb-0.1"
    },

    "2": {
        "pretest": "shared",
        "experimentA": "nfb-10",
        "experimentB": "vfb-0.1",
        "posttest": "shared",
        "unstable_example": "shared",
        "stable_example": "shared",
        "mass_example": "nfb-10"
    },

    "3": {
        "pretest": "shared",
        "experimentA": "nfb-10",
        "experimentB": "vfb-10",
        "posttest": "shared",
        "unstable_example": "shared",
        "stable_example": "shared",
        "mass_example": "nfb-10"
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
    if len(meta) == 0:
        raise RuntimeError("could not find metadata for '%s' with k=%s" % (
            stim, kappa))
    if len(meta) > 1:
        assert (meta['nfell'][0] == meta['nfell']).all()
        assert (meta['stable'][0] == meta['stable']).all()
    return meta[:1]


for condition, maps in conditions.iteritems():
    for cb in [0, 1]:
        config = {}
        for phase, pth in maps.iteritems():

            if phase in ("pretest", "posttest"):
                render_config = render_dir.joinpath(
                    pth, "%s.csv" % phase)

                fb = "vfb"
                ratio = "1"
                ask_fall = True
                ask_mass = False

            elif phase in ("stable_example", "unstable_example"):
                render_config = render_dir.joinpath(
                    pth, "%s.csv" % phase)

                fb = "vfb"
                ratio = "1"
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

            # XXX: hack! kappa might not be correct for the mass
            # example because we want the example tower to be the same
            # for everyone (including whether it falls or not), so for
            # counterbalanced trials we actually flip kappa rather
            # than the colors
            if phase == "mass_example":
                conf.kappa = np.log10(float(ratio))

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

            r2kappa = np.array(map(float, conf.ratio))
            assert (10**conf.kappa == r2kappa).all()

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

        configpath = json_dir.joinpath("%s-cb%d.json" % (condition, cb))
        with open(configpath, "w") as fh:
            json.dump(config, fh, indent=2, allow_nan=False)
