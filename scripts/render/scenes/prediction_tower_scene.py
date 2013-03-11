import cogphysics
from cogphysics.core.graphics import PandaGraphics as Graphics
from tower_scene_base import TowerScene

import panda3d.core as p3d
import pandac.PandaModules as pm
import libpanda as lp

import numpy as np
import os


class PredictionTowerScene(TowerScene):

    def setBlockProperties(self, kappa=None, mu=None):
        rand = np.random.RandomState(0)
        lcscale = 0.15
        hcscale = 0.1
        clip = lambda x: np.clip(x, 0, 1)

        try:
            strparams = self.scene.label.split("~")[1].split("_")
            paramdict = dict([x.split("-", 1) for x in strparams])
        except:
            paramdict = {}

        if not kappa and 'kappa' in paramdict:
            kappa = float(paramdict['kappa'])
        elif not kappa:
            print "Warning: no kappa specified, defaulting to 1:1 ratio"
            kappa = 0.0

        d0 = 170
        d1 = 170 * (10 ** kappa)
        print "kappa is", kappa
        self.kappa = kappa

        if not mu:
            mu = 0.8
        surface = "mass_tower_%02d" % int(mu * 10)

        for bidx, block in enumerate(self.blocks):
            # friction setting
            block.surface = surface

            block.model = 'block'
            r = rand.randn()*lcscale/2.0

            # blocks of type 0 are light (plastic)
            if block.meta['type'] == 0:
                block.color = (
                    clip(.3 + r),
                    clip(1. + rand.randn()*lcscale/2. + r),
                    clip(.65 + rand.randn()*lcscale/2. + r),
                    1)
                if d0 is not None:
                    block.density = d0

            # blocks of type 1 are heavy (stone)
            elif block.meta['type'] == 1:
                block.color = (
                    clip(.65 + rand.randn()*hcscale/2. + r),
                    clip(.65 + r),
                    clip(.65 + rand.randn()*hcscale/2. + r),
                    1)
                if d1 is not None:
                    block.density = d1

            # invalid type
            else:
                print "Warning: bad block type"

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

            if block.meta['type'] == 0:
                tex = loader.loadTexture(
                    os.path.join(texpath, 'stripes.png'))
                scl = np.max(block.scale) / 5
                hpr = (
                    rand.rand()*180,
                    rand.rand()*180,
                    rand.rand()*180)
                offset = (
                    rand.rand()*scl*2 - scl,
                    rand.rand()*scl*2 - scl)
                scale = (2.0, 2.0)

            elif block.meta['type'] == 1:
                tex = loader.loadTexture(
                    os.path.join(texpath, 'granite-grayscale.jpg'))
                scl = np.max(block.scale) / 10
                hpr = (
                    rand.rand()*20,
                    rand.rand()*20,
                    rand.rand()*20)
                offset = (
                    rand.rand()*scl*2 - scl,
                    rand.rand()*scl*2 - scl)
                scale = (
                    rand.rand()*1.5+.5,
                    rand.rand()*1.5+.5)

            ts = lp.TextureStage('ts_%s' % block.label)
            block.graphics.node.setTexture(ts, tex, 1)
            block.graphics.node.setTexHpr(ts, *hpr)
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
