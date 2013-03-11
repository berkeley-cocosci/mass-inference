import cogphysics
from cogphysics.core.graphics import PandaGraphics as Graphics
from tower_scene_base import TowerScene

import panda3d.core as p3d
import pandac.PandaModules as pm
import libpanda as lp

import numpy as np
import os
import random
import colorsys


class OriginalTowerScene(TowerScene):

    def setBlockProperties(self, kappa=None, mu=None):
        if not mu:
            mu = 0.8
        surface = "mass_tower_%02d" % int(mu * 10)

        for bidx, block in enumerate(self.blocks):
            # friction setting
            block.surface = surface
            # model
            block.model = 'block'
            # color
            (r, g, b) = colorsys.hsv_to_rgb(random.random(), 1, 1)
            block.color = (r, g, b, 1.0)

        self.scene.resetInit(fchildren=True)

    def setGraphics(self):
        self.scene.graphics = Graphics
        self.scene.enableGraphics()
        self.scene.propagate('graphics')
        texpath = cogphysics.path(cogphysics.TEXTURE_PATH, 'local')
        rand = np.random.RandomState(1)

        for block in self.blocks:
            block.graphics.node.clearMaterial()
            block.graphics.node.clearTexture()
            block.graphics.node.setScale((0.1, 0.1, 0.1))
            # block.graphics.node.setDepthOffset(1)

            tex = loader.loadTexture(
                os.path.join(texpath, 'noise.rgb'))
            scl = np.max(block.scale) / 5
            hpr = (
                rand.rand()*180,
                rand.rand()*180,
                rand.rand()*180)
            offset = (
                rand.rand()*scl*2 - scl,
                rand.rand()*scl*2 - scl)
            scale = (2.0, 2.0)

            ts = lp.TextureStage('ts_%s' % block.label)
            block.graphics.node.setTexture(ts, tex, 1)
            #block.graphics.node.setTexHpr(ts, *hpr)
            block.graphics.node.setTexOffset(ts, *offset)
            block.graphics.node.setTexScale(ts, *scale)

            for mat in block.graphics.node.findAllMaterials():
                mat.clearDiffuse()
                mat.clearAmbient()
                mat.setShininess(10)
                mat.setSpecular((0.2, 0.2, 0.2, 1.0))

        for mat in self.table.graphics.node.findAllMaterials():
            mat.setShininess(0)
            mat.clearSpecular()
            mat.clearAmbient()
            mat.clearDiffuse()
        # self.table.graphics.node.setDepthOffset(1)
