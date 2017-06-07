#!/usr/bin/env python

import json
import os
from scenesim.objects.sso import SSO
from libpanda import LVector3f, LPoint3f, LVecBase3f, LQuaternionf, LVecBase4f

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if not os.path.exists(os.path.join(root, "resources/sso_json")):
    os.makedirs(os.path.join(root, "resources/sso_json"))

sso_paths = []
for dirname, dirnames, filenames in os.walk(os.path.join(root, "resources/sso")):
    for filename in filenames:
        if filename.endswith(".cpo"):
            sso_paths.append(os.path.join(dirname, filename))

for filename in sso_paths:
    sso_name = os.path.splitext(os.path.basename(filename))[0]
    dest = os.path.join(os.path.split(filename)[0], "{}.json".format(sso_name))
    dest = dest.replace("/sso/", "/sso_json/")
    print(dest)

    if not os.path.exists(os.path.dirname(dest)):
        os.makedirs(os.path.dirname(dest))

    sso = SSO.load_tree(filename)
    types, props, porder = sso.state_prop()
    types = [[t.__module__, t.__name__] for t in types]
    for p in props:
        for prop, val in p.items():
            if isinstance(val, (LVecBase3f, LVecBase4f, LVector3f, LPoint3f, LQuaternionf)):
                p[prop] = {"__class__": [type(val).__module__, type(val).__name__], "value": list(val)}

    with open(dest, 'w') as fh:
        json.dump([types, props, porder], fh)
