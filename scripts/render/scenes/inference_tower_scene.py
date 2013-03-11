import cogphysics
from cogphysics.core.graphics import PandaGraphics as Graphics
from tower_scene_base import TowerScene

import panda3d.core as p3d
import pandac.PandaModules as pm
import libpanda as lp

import numpy as np
import os


class InferenceTowerScene(TowerScene):

    def setBlockProperties(self, kappa=None, mu=None, counterbalance=None):
        if counterbalance is None:
            counterbalance = False

        strparams = self.scene.label.split("~")[1].split("_")
        paramdict = dict([x.split("-", 1) for x in strparams])

        if not kappa and 'kappa' in paramdict:
            kappa = float(paramdict['kappa'])
        if kappa:
            d0 = 170
            d1 = 170 * (10 ** kappa)
        else:
            d0 = d1 = None
        self.kappa = kappa
        print "kappa is", kappa

        if not mu:
            mu = 0.8
        surface = "mass_tower_%02d" % int(mu * 10)

        type0 = 'red_block' if not counterbalance else 'yellow_block'
        color0 = (1, 0, 0, 1) if not counterbalance else (1, 1, 0, 1)
        type1 = 'yellow_block' if not counterbalance else 'red_block'
        color1 = (1, 1, 0, 1) if not counterbalance else (1, 0, 0, 1)

        for bidx, block in enumerate(self.blocks):
            # friction setting
            block.surface = surface

            # set density according to the counterbalance
            if block.meta['type'] == 0:
                block.model = type0
                block.color = color0
                if d0 is not None:
                    block.density = d0

            elif block.meta['type'] == 1:
                block.model = type1
                block.color = color1
                if d1 is not None:
                    block.density = d1

            # invalid type
            else:
                raise ValueError("bad block type")

        self.scene.resetInit(fchildren=True)

    def setGraphics(self):
        self.scene.graphics = Graphics
        self.scene.enableGraphics()
        self.scene.propagate('graphics')

        if self.kappa == 0.0 or self.kappa is None:
            texpath = cogphysics.path(cogphysics.TEXTURE_PATH, 'local')
            rand = np.random.RandomState(1)

            for block in self.blocks:
                block.graphics.node.clearMaterial()
                block.graphics.node.clearTexture()
                block.graphics.node.setScale((0.1, 0.1, 0.1))

                tex = loader.loadTexture(
                    os.path.join(texpath, 'noise.rgb'))
                scl = np.max(block.scale) / 5
                offset = (
                    rand.rand()*scl*2 - scl,
                    rand.rand()*scl*2 - scl)
                scale = (2.0, 2.0)

                ts = lp.TextureStage('ts_%s' % block.label)
                block.graphics.node.setTexture(ts, tex, 1)
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
