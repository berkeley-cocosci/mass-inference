from view_towers_base import ViewTowers
from scenes import InferenceTowerScene
from arghelper import parseargs

import os
import pickle

INFOPATH = "../../stimuli/meta"

class ViewOneshotTowers(ViewTowers):

    towerscene_type = InferenceTowerScene

    def __init__(self, cpopath, playback):
        self.stiminfo = {}

        ViewTowers.__init__(self, cpopath, playback)

        self.accept('[', self.decreaseMass)
        self.accept(']', self.increaseMass)

    def changeMass(self, newkappa):
        if self.playback:
            scene = self.towerscene.scene.label
            cponame, strparams = scene.split("~")
            params = [x.split("-", 1) for x in strparams.split("_")]
            for idx in xrange(len(params)):
                if params[idx][0] == 'kappa':
                    params[idx][1] = str(newkappa)
            strparams = "_".join(["-".join(x) for x in params])
            scene = "%s~%s" % (cponame, strparams)
            self.loadScene(scene)
        else:
            self.reset()
            self.towerscene.setBlockProperties(kappa=newkappa)

    def increaseMass(self):
        kappa = self.towerscene.kappa
        if self.playback:
            if kappa == -1.0:
                # hack
                kappa = 1.0
            else:
                return
        else:
            kappa = self.towerscene.kappa + 0.1
        self.changeMass(kappa)

    def decreaseMass(self):
        kappa = self.towerscene.kappa
        if self.playback:
            if kappa == 1.0:
                # hack
                kappa = -1.0
            else:
                return
        else:
            kappa = self.towerscene.kappa - 0.1
        self.changeMass(kappa)

    def setWireframeColors(self):
        # set block wireframe colors
        for block in self.towerscene.blocks:
            block.meta['color'] = block.color

    def unsetWireframeColors(self):
        # set the block colors back to normal
        for idx, block in enumerate(self.towerscene.blocks):
            block.color = block.meta['color']

    def loadScene(self, scene, cpopath=None):
        ViewTowers.loadScene(self, scene, cpopath=None)
        if scene is not None:
            stiminfo = self.stiminfo.get(scene + "_cb-0", {})

            color0 = stiminfo.get('color0', None)
            color1 = stiminfo.get('color1', None)
            if color0 and color1:
                print "Colors are: %s, %s" % (color0, color1)
                self.towerscene.setBlockProperties(
                    color0=color0, color1=color1)
                self.towerscene.setGraphics()

if __name__ == "__main__":
    scenes, cpopath, cam_start, playback = parseargs()

    infopath = os.path.join(INFOPATH, "F-stimulus-info.pkl")
    with open(infopath, "r") as fh:
        stiminfo = pickle.load(fh)

    app = ViewOneshotTowers(cpopath, playback)
    app.scenes = scenes
    app.stiminfo = stiminfo
    app.cam_start = cam_start
    app.rotating.setH(app.cam_start)
    app.sidx = -1
    app.nextScene()
    app.run()
