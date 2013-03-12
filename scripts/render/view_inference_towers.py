from view_towers_base import ViewTowers
from scenes import RYTowerScene
from arghelper import parseargs


class ViewInferenceTowers(ViewTowers):

    towerscene_type = RYTowerScene

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
