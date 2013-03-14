from view_towers_base import ViewTowers
from scenes import RYTowerScene
from arghelper import parseargs


class ViewInferenceTowers(ViewTowers):

    towerscene_type = RYTowerScene

    def __init__(self, cpopath, playback):
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


if __name__ == "__main__":
    scenes, cpopath, cam_start, playback = parseargs()
    app = ViewInferenceTowers(cpopath, playback)
    app.scenes = scenes
    app.cam_start = cam_start
    app.rotating.setH(app.cam_start)
    app.sidx = -1
    app.nextScene()
    app.run()
