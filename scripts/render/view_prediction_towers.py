from view_towers_base import ViewTowers
from scenes import PredictionTowerScene
from arghelper import parseargs


class ViewPredictionTowers(ViewTowers):

    towerscene_type = PredictionTowerScene

    def setWireframeColors(self):
        # set block wireframe colors
        for block in self.towerscene.blocks:
            block.meta['color'] = block.color
            if block.meta['type'] == 0:
                block.color = (0, 0.7, 0, 1)
            elif block.meta['type'] == 1:
                block.color = (0, 0, 0, 1)

    def unsetWireframeColors(self):
        # set the block colors back to normal
        for idx, block in enumerate(self.towerscene.blocks):
            block.color = block.meta['color']


if __name__ == "__main__":
    scenes, cpopath, cam_start, playback = parseargs()
    app = ViewPredictionTowers(cpopath, playback)
    app.scenes = scenes
    app.cam_start = cam_start
    app.rotating.setH(app.cam_start)
    app.sidx = -1
    app.nextScene()
    app.run()
