import cogphysics
import cogphysics.lib.funclib as fl
import cogphysics.lib.geometry as geom
import cogphysics.lib.physutil as physutil

from cogphysics.core import cpObject
from cogphysics.core.graphics import PandaGraphics as Graphics
from cogphysics.core.physics import OdePhysics as Physics
from cogphysics.mass.towers.tower_scene import TowerScene

import panda3d.core as p3d
import pandac.PandaModules as pm
import libpanda as lp
from direct.gui.OnscreenText import OnscreenText, TextNode
from direct.showbase.ShowBase import ShowBase

import datetime
import numpy as np
import os
import pdb
import random
import sys

class PredictionTowerScene(TowerScene):

    # def __init__(self):
    #     super(TowerScene, self).__init__()
    #     self.cpopath = cogphysics.path(os.path.join(
    #         cogphysics.CPOBJ_PATH, "mass", "new-stability-towers"), 'local', 'afs')
    #     super(TowerScene, self).cpopath = cogphysics.path(os.path.join(
    #         cogphysics.CPOBJ_PATH, "mass", "new-stability-towers"), 'local', 'afs')
        
    def setBlockProperties(self, kappa=None, mu=None):
        rand = np.random.RandomState(0)
        lcscale = 0.15
        hcscale = 0.1
        clip = lambda x: np.clip(x, 0, 1)
        
        if kappa:
            d0 = 170
            d1 = 170 * (10 ** kappa)
        else:
            d0 = d1 = None

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
                block.color = (clip(.3  + r),
                               clip(1.  + rand.randn()*lcscale/2. + r),
                               clip(.65 + rand.randn()*lcscale/2. + r),
                               1)
                if d0 is not None:
                    block.density = d0

            # blocks of type 1 are heavy (stone)
            elif block.meta['type'] == 1:
                block.color = (clip(.65 + rand.randn()*hcscale/2. + r),
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

            # for tex in block.graphics.node.findAllTextures():
            #     tex.clear()
                
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

            # For the transparent blocks -- you get this weird occluding
            # flickering where it will draw the block but not the objects behind
            # it (even though you should be able to see them, because it's
            # transparent).
            #
            # We need to set the depth write to false (so the transparent object
            # doesn't affect the depth buffer, i.e. the back of the block won't
            # be occluded by the front of the block)
            #
            # We also need to set the sort bin to 'fixed' (so the transparent
            # blocks are drawn after all other objects).
            #
            # We use two nodes so that we can fill the depth buffer first (for
            # correct shading), and then the color buffer later.  See:
            #
            # http://www.panda3d.org/forums/viewtopic.php?t=5720
            # if block.meta['type'] == 0:
            #     block.graphics.node2 = block.graphics.node.copyTo(
            #         block.graphics.node.getParent())

            #     block.graphics.node2.setAttrib(
            #         lp.ColorWriteAttrib.make(lp.ColorWriteAttrib.COff))
            #     block.graphics.node2.setDepthWrite(False)

            #     block.graphics.node2.setBin('fixed', 0)
            #     block.graphics.node.setBin('fixed', 10)

        for mat in self.table.graphics.node.findAllMaterials():
            mat.setShininess(0)
            mat.clearSpecular()
            mat.clearAmbient()
            mat.clearDiffuse()
