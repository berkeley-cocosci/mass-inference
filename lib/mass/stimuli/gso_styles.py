import numpy as np
from libpanda import TextureStage
from util import get_rgb


class GSOStyler(object):

    def __init__(self, loader):
        self.loader = loader

    def _rso(self, gso):
        # use a random seed based on the block's name, so we can have
        # "random" properties that are actually always the same for
        # any given block
        rso = np.random.RandomState(abs(hash(repr(gso))))
        return rso

    def floor(self, gso, **kwargs):
        gso.setColor((0.3, 0.3, 0.3, 1))

    def original(self, gso, **kwargs):
        gso.destroy_resources(tags=("model",))
        gso.set_model("wood_block")
        gso.init_resources(tags=("model",))
        gso.setScale(0.5, 1./6., 0.5)

    def mass_plastic_stone(self, gso, **kwargs):
        blocktype = kwargs['blocktype']

        rso = self._rso(gso)
        hpr = rso.rand(3)
        offset = rso.rand(2)
        scale = rso.rand(2)

        if blocktype == 0:
            tex = self.loader.loadTexture('stripes.png')
            scl = np.max(gso.getScale()) / 5.
            hpr *= 180
            offset = offset*scl*2 - scl
            scale = (2.0, 2.0)

        elif blocktype == 1:
            gso.destroy_resources(tags=("model",))
            gso.set_model("stone_block")
            gso.init_resources(tags=("model",))
            gso.setScale(0.5, 1./6., 0.5)

            for mat in gso.findAllMaterials():
                mat.setShininess(0)
                mat.setSpecular((0, 0, 0, 0))

            tex = self.loader.loadTexture('granite-grayscale.jpg')
            scl = np.max(gso.getScale()) / 10.
            hpr *= 20
            offset = offset*scl*2 - scl
            scale = scale*1.5 + 0.5

        ts = TextureStage('ts_%s' % gso.getName())
        gso.setTexture(ts, tex, 1)
        gso.setTexHpr(ts, *hpr)
        gso.setTexOffset(ts, *offset)
        gso.setTexScale(ts, *scale)

    def mass_red_yellow(self, gso, **kwargs):
        pass

    def mass_colors(self, gso, **kwargs):
        blocktype = kwargs['blocktype']
        color0 = kwargs['color0']
        color1 = kwargs['color1']

        rso = self._rso(gso)

        gso.destroy_resources(tags=("model",))
        gso.set_model("wood_block")
        gso.init_resources(tags=("model",))
        gso.setScale(0.5, 1./6., 0.5)

        if blocktype == 0 and color0:
            gso.setColor(get_rgb(color0, rso))
        elif blocktype == 1 and color1:
            gso.setColor(get_rgb(color1, rso))

    def apply(self, gso, style, **kwargs):
        style_func = getattr(self, style)
        style_func(gso, **kwargs)
