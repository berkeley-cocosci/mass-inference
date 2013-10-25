import dbtools
import json
import numpy as np
import pandas as pd
from path import path

conditions = {
    "0": {
        "pretest": "shared",
        "experimentA": "vfb-0.1",
        "experimentB": "nfb-0.1",
        "experimentC": "vfb-0.1",
        "posttest": "shared",
        "unstable_example": "shared",
        "stable_example": "shared",
        "mass_example": "vfb-0.1"
    },

    "1": {
        "pretest": "shared",
        "experimentA": "vfb-0.1",
        "experimentB": "nfb-0.1",
        "experimentC": "vfb-10",
        "posttest": "shared",
        "unstable_example": "shared",
        "stable_example": "shared",
        "mass_example": "vfb-0.1"
    },

    "2": {
        "pretest": "shared",
        "experimentA": "vfb-10",
        "experimentB": "nfb-10",
        "experimentC": "vfb-0.1",
        "posttest": "shared",
        "unstable_example": "shared",
        "stable_example": "shared",
        "mass_example": "vfb-10"
    },

    "3": {
        "pretest": "shared",
        "experimentA": "vfb-10",
        "experimentB": "nfb-10",
        "experimentC": "vfb-10",
        "posttest": "shared",
        "unstable_example": "shared",
        "stable_example": "shared",
        "mass_example": "vfb-10"
    },
}

render_dir = path("../experiment/static/stimuli/")
json_dir = path("../experiment/static/json/")

with open(json_dir.joinpath("conditions.json"), "w") as fh:
    json.dump(conditions, fh, indent=2, allow_nan=False)

DBPATH = path("../resources/sso/metadata.db")
tbl = dbtools.Table(DBPATH, "stability")


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
            render_config = render_dir.joinpath(
                "%s-cb%d" % (pth, cb), "%s.csv" % phase)

            if phase in ("pretest", "posttest"):
                fb = "vfb"
                ratio = "1"

            elif phase in ("stable_example", "unstable_example"):
                fb = "vfb"
                ratio = "1"

            # XXX: hack! ratio might not be correct for the mass
            # example because we want the example tower to be the same
            # for everyone (including whether it falls or not), so we
            # don't actually use a different kappa
            elif phase == "mass_example":
                fb = "vfb"
                ratio = "10"

            else:
                fb, ratio = pth.split("-")

            print condition, cb, phase

            conf = pd.DataFrame.from_csv(render_config).reset_index()
            conf.stimulus = conf.stimulus.map(lambda x: path(x).namebase)

            conf['feedback'] = fb
            conf['ratio'] = ratio
            conf['counterbalance'] = cb

            meta = pd.concat(map(get_meta, conf.stimulus, conf.kappa))
            meta['stable'] = meta['stable'].astype('bool')
            meta = meta.drop('dataset', axis=1)

            conf = pd.merge(conf, meta, on=['stimulus', 'kappa'])
            conf = conf.set_index('stimulus').sort().reset_index()

            r2kappa = np.array(map(float, conf.ratio))
            assert (10**conf.kappa == r2kappa).all()

            trials = conf.reset_index(drop=True).T.to_dict().values()
            trials.sort(cmp=lambda x, y: cmp(x['stimulus'], y['stimulus']))
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
