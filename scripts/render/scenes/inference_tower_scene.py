import os
import pdb
import numpy as np

import cogphysics
from cogphysics.core.graphics import PandaGraphics as Graphics
from tower_scene_base import TowerScene

import panda3d.core as p3d
import pandac.PandaModules as pm
import libpanda as lp


def convertColor(color):
    assert color[0] == "#"
    r = int(color[1:3], base=16) / 255.
    g = int(color[3:5], base=16) / 255.
    b = int(color[5:7], base=16) / 255.
    colortup = (r, g, b, 1)
    return colortup


class InferenceTowerScene(TowerScene):

    def setBlockProperties(self, kappa=None, color0=None, color1=None,
                           counterbalance=None):
        try:
            strparams = self.scene.label.split("~")[1].split("_")
            paramdict = dict([x.split("-", 1) for x in strparams])
        except:
            paramdict = {}

        if counterbalance is None and 'cb' in paramdict:
            counterbalance = bool(int(paramdict['cb']))
        elif counterbalance is None:
            print("Warning: no counterbalance specified, "
                  "so not counterbalancing")
            counterbalance = False

        if not kappa and 'kappa' in paramdict:
            kappa = float(paramdict['kappa'])
        elif not kappa:
            print "Warning: no kappa specified, defaulting to 1:1 ratio"
            kappa = 0.0

        d0 = 170
        d1 = 170 * (10 ** kappa)
        print "kappa is", kappa
        self.kappa = kappa

        if not color0 or not color1:
            print("Warning: both block colors not specified, defaulting to "
                  "red and yellow")
            color0 = "#FF0000"
            color1 = "#FFFF00"

        color0 = convertColor(color0)
        color1 = convertColor(color1)
        cbcolor0 = color0 if not counterbalance else color1
        cbcolor1 = color1 if not counterbalance else color0

        mu = 0.8
        surface = "mass_tower_%02d" % int(mu * 10)

        for bidx, block in enumerate(self.blocks):
            # friction setting
            block.surface = surface
            # model
            block.model = 'block'

            # set density according to the counterbalance
            if block.meta['type'] == 0:
                block.color = cbcolor0
                if d0 is not None:
                    block.density = d0

            elif block.meta['type'] == 1:
                block.color = cbcolor1
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
        texpath = cogphysics.path(cogphysics.TEXTURE_PATH, 'local')
        rand = np.random.RandomState(1)

        for block in self.blocks:
            block.graphics.node.clearMaterial()
            block.graphics.node.clearTexture()
            block.graphics.node.setScale((0.1, 0.1, 0.1))

            tex = loader.loadTexture(
                os.path.join(texpath, 'noisy-grayscale.jpg'))
                # os.path.join(texpath, 'noise.rgb'))
            scl = np.max(block.scale) / 10
            hpr = (
                rand.rand()*20,
                rand.rand()*20,
                rand.rand()*20)
            offset = (
                rand.rand()*scl*2 - scl,
                rand.rand()*scl*2 - scl)
            # scale = (
            #     rand.rand()*2.5+.5,
            #     rand.rand()*2.5+.5)
            scale = (0.2, 0.2)
            # scl = np.max(block.scale) / 5
            # hpr = (
            #     rand.rand()*180,
            #     rand.rand()*180,
            #     rand.rand()*180)
            # offset = (
            #     rand.rand()*scl*2 - scl,
            #     rand.rand()*scl*2 - scl)
            # scale = (2.0, 2.0)

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
