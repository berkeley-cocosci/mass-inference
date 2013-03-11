import os
import datetime

from optparse import OptionParser

from cogphysics.core.physics import OdePhysics as Physics
from cogphysics.lib.physutil import recordPlayback

from prediction_tower_scene import PredictionTowerScene
from original_tower_scene import OriginalTowerScene

PBPATH = "../../data/playback"
CPOPATH = "../../stimuli/obj/old"
LISTPATH = "../../stimuli/lists"

time = 3
physStep = 1. / 1000.
physStride = 1


def record(stype, lists, original=False):

    #pbpath = os.path.join(PBPATH, stype)
    pbpath = PBPATH
    cpopath = os.path.join(CPOPATH, stype)

    cpos = []
    for l in lists:
        lp = os.path.join(LISTPATH, l)
        with open(lp, "r") as fh:
            cpos.extend(fh.read().strip().split("\n"))

    for cpo in cpos:

        if original:
            cponame = cpo
            kappa = 0.0
        else:
            c, k = cpo.split("~")
            kappa = float(k[len("kappa-"):])
            cponame = '%s~kappa-%.1f' % (c, kappa)
            assert cpo == cponame

        if os.path.exists(os.path.join(pbpath, cponame + ".pb.npz")):
            print "%s exists, skipping..." % cponame
            continue

        starttime = datetime.datetime.now()

        # load the cpo
        if original:
            towerscene = OriginalTowerScene.create(cponame, cpopath=cpopath)
        else:
            towerscene = PredictionTowerScene.create(cponame, cpopath=cpopath)
        towerscene.setBlockProperties(kappa=kappa)
        scene = towerscene.scene

        # create physics
        scene.physics = Physics
        scene.enablePhysics()
        scene.propagate('physics')

        # record positions
        recordPlayback(
            scene.getDescendants(fself=True),
            time,
            physstep=physStep,
            stride=physStride,
            forces=[])
        scene.pbSave(cponame, pbpath, fchildren=True)

        # destroy physics and the scene
        scene.physics = None
        scene.propagate('physics')
        scene.destroy()

        timediff = (datetime.datetime.now() - starttime).total_seconds()
        print "%s : %s sec" % (cponame, timediff)


if __name__ == "__main__":
    usage = "usage: %prog [options] list1 [list2 ... listN]"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "-s", "--stype", dest="stype",
        help="stimulus type, e.g. mass-learning [required]",
        metavar="STIM_TYPE")
    parser.add_option(
        "-o", "--original",
        action="store_true", dest="original", default=False,
        help="stimuli are original stability (non-mass) towers")

    (options, args) = parser.parse_args()
    if len(args) == 0:
        print("Warning: no stimuli lists passed. "
              "Assuming '%s' instead." % options.stype)
        args = [options.stype]
    record(options.stype, args, original=options.original)
