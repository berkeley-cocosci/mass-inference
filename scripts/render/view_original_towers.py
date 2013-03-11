import sys

from view_towers_base import ViewTowers
from scenes import OriginalTowerScene
from arghelper import parseargs


class ViewOriginalTowers(ViewTowers):

    towerscene_type = OriginalTowerScene

    def setWireframeColors(self):
        for block in self.towerscene.blocks:
            block.meta['color'] = block.color
            block.color = (1, 1, 1, 1)

    def unsetWireframeColors(self):
        for idx, block in enumerate(self.towerscene.blocks):
            block.color = block.meta['color']


if __name__ == "__main__":
    scenes, cpopath, cam_start, playback = parseargs()
    app = ViewOriginalTowers(cpopath, playback)
    app.scenes = scenes
    app.cam_start = cam_start
    app.rotating.setH(app.cam_start)
    app.sidx = -1
    app.nextScene()
    app.run()
