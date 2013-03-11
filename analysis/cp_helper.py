"""Cogphysics utility functions"""

import cogphysics
import cogphysics.lib.hashtools as ht

import os
import pickle
import shutil


def convert(x, name_table, conv_inv):
    try:
        n = name_table[x]
    except KeyError:
        pass
    else:
        return n
    try:
        n = name_table[ht.reverse_find_hashes(conv_inv[x])[-1]]
    except KeyError:
        pass
    else:
        return n

    try:
        n = name_table[ht.forward_find_hashes(conv_inv[x])[-1]]
    except KeyError:
        pass
    else:
        return n

    return None


def copy_stims(target, source):

    srcnewdir = os.path.join(cogphysics.STIM_PATH, source)
    if "tower_mass" in source:
        srcolddir = os.path.join(cogphysics.CPOBJ_PATH, "mass", "all-towers")
    else:
        srcolddir = os.path.join(cogphysics.CPOBJ_PATH, "stability")

    dstnewdir = os.path.join("../stimuli/obj/new", target)
    dstolddir = os.path.join("../stimuli/obj/old", target)
    if not os.path.exists(dstnewdir):
        os.makedirs(dstnewdir)
    if not os.path.exists(dstolddir):
        os.makedirs(dstolddir)

    name_table_path = os.path.join(srcnewdir, "name_table.pkl")
    with open(name_table_path, "r") as fid:
        name_table = dict(pickle.load(fid))

    pth = os.path.join(cogphysics.RESOURCE_PATH, 'cpobj_conv_stability.pkl')
    with open(pth, "r") as fh:
        conv = pickle.load(fh)
    conv_inv = dict([(x[1], x[0]) for x in conv.items()])

    pth = os.path.join("../stimuli/lists", target)
    if "tower_mass" in source:
        pth += "~kappa-1.0"
    with open(pth, "r") as fh:
        all_stim = fh.read().split('\n')
    all_stim = [x.split("~")[0] for x in all_stim if x != '']
    newnames = [convert(x, name_table, conv_inv) for x in all_stim]
    newnames = [x + '.cpo' if not x.endswith('.cpo') else x for x in newnames]

    new_name_table1 = dict(zip(all_stim, newnames))
    new_name_table2 = dict(zip(newnames, all_stim))

    for name1, name2 in zip(newnames, all_stim):
        srcpathnew = os.path.join(srcnewdir, name1)
        dstpathnew = os.path.join(dstnewdir, name1)
        srcpathold = os.path.join(srcolddir, name2)
        dstpathold = os.path.join(dstolddir, name2)
        shutil.copy(srcpathnew, dstpathnew)
        shutil.copy(srcpathold, dstpathold)
        print name1, "<-->", name2
        print srcpathnew, "-->", dstpathnew
        print srcpathold, "-->", dstpathold

    with open(os.path.join(dstnewdir, 'name_table.pkl'), "w") as fh:
        pickle.dump(new_name_table1, fh)
    with open(os.path.join(dstolddir, 'name_table.pkl'), "w") as fh:
        pickle.dump(new_name_table2, fh)

    srcpathnew = os.path.join(srcnewdir, 'floor.cpo')
    dstpathnew = os.path.join(dstnewdir, 'floor.cpo')
    shutil.copy(srcpathnew, dstpathnew)
