#!/usr/bin/env python

import json
import os
import pandas as pd


def unicode_to_str(obj):
    if isinstance(obj, dict):
        obj = {str(k): unicode_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        obj = [unicode_to_str(x) for x in obj]
    elif isinstance(obj, unicode):
        obj = str(obj)

    return obj


root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sso_paths = {}
for dirname in sorted(os.listdir(os.path.join(root, "resources", "sso_json"))):
    sso_paths[dirname] = []
    for filename in sorted(os.listdir(os.path.join(root, "resources", "sso_json", dirname))):
        if filename.endswith(".json"):
            sso_paths[dirname].append(os.path.join(root, "resources", "sso_json", dirname, filename))

for dirname in sso_paths:
    ssos = []
    for filename in sso_paths[dirname]:
        sso_name = os.path.splitext(os.path.basename(filename))[0]
        print(sso_name)

        with open(filename, 'r') as fh:
            types, props, porder = json.load(fh)

        props = unicode_to_str(props)
        newprops = []
        for i, p in list(enumerate(props))[::-1]:
            for prop, val in p.items():
                if isinstance(val, dict) and "__class__" in val:
                    p[prop] = val["value"]

            if p["name"].endswith("physics"):
                p["object"] = p["name"].rsplit("-", 1)[0]
                p["name"] = sso_name

                for j, dim in enumerate(["x", "y", "z"]):
                    p["scale_{}".format(dim)] = p["scale"][j]
                    p["pos_{}".format(dim)] = p["pos"][j]
                del p["scale"]
                del p["pos"]

                for j, dim in enumerate(["w", "x", "y", "z"]):
                    p["quat_{}".format(dim)] = p["quat"][j]
                del p["quat"]

                del p["deactivation_enabled"]
                del p["linear_velocity"]
                del p["gravity"]
                del p["angular_velocity"]
                del p["into_collide_mask"]
                newprops.append(p)

        ssos.extend(newprops)

    df = pd.DataFrame(ssos).set_index(["name", "object"]).sortlevel()
    dest = os.path.join(root, "resources", "sso", "{}.csv".format(dirname))
    df.to_csv(dest)
    print(dest)
