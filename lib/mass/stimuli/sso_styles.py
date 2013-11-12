import colorsys
import numpy as np
from libpanda import Vec4
from scenesim.objects.gso import GSO
from scenesim.objects.pso import RBSO

from util import get_blocktypes


class SSOStyler(object):

    def _rso(self, sso):
        rso = np.random.RandomState(abs(hash(repr(sso))))
        return rso

    def floor(self, sso):
        """Settings for round wooden floor SSO."""
        gsos = sso.descendants(type_=GSO)
        assert len(gsos) == 1
        gso = gsos[0]
        gso.set_model("wood_floor.egg")
        gso.setColor((0.45, 0.3, 0.1, 1.0))
        sso.set_friction(0.8 ** 0.5)
        sso.set_restitution(0)
        sso.setScale((10, 10, 1))

    def _block_physics(self, pso):
        """Physics settings for tower blocks."""
        density = 170.0
        volume = np.prod(pso.getScale())
        mass = density * volume
        pso.set_mass(mass)
        pso.set_friction(0.8 ** 0.5)
        pso.set_restitution(0)

    def original(self, sso):
        """Settings for original-type towers (random colors)."""
        rso = self._rso(sso)

        psos = sso.descendants(type_=RBSO)
        gsos = sso.descendants(type_=GSO)
        for pso, gso in zip(psos, gsos):
            r, g, b = colorsys.hsv_to_rgb(rso.rand(), 1, 1)
            color = Vec4(r, g, b, 1)

            gso.setColor(color)
            gso.set_model("block.egg")
            self._block_physics(pso)

    def mass_plastic_stone(self, sso):
        """Settings for prediction mass towers (plastic/stone blocks)."""
        rso = self._rso(sso)

        lcscale = 0.15
        hcscale = 0.1
        clip = lambda x: np.clip(x, 0, 1)

        psos = sso.descendants(type_=RBSO)
        gsos = sso.descendants(type_=GSO)
        blocktypes = get_blocktypes(sso)

        for gso, pso, blocktype in zip(gsos, psos, blocktypes):
            r = rso.randn()*lcscale/2.0
            if blocktype == 0:
                color = Vec4(
                    clip(.3 + r),
                    clip(1. + rso.randn()*lcscale/2. + r),
                    clip(.65 + rso.randn()*lcscale/2. + r),
                    1)
            elif blocktype == 1:
                color = Vec4(
                    clip(.65 + rso.randn()*hcscale/2. + r),
                    clip(.65 + r),
                    clip(.65 + rso.randn()*hcscale/2. + r),
                    1)

            gso.setColor(color)
            gso.set_model("block.egg")
            self._block_physics(pso)

    def mass_red_yellow(self, sso):
        """Settings for RY inference mass towers (red/yellow blocks)."""
        psos = sso.descendants(type_=RBSO)
        gsos = sso.descendants(type_=GSO)
        blocktypes = get_blocktypes(sso)

        for gso, pso, blocktype in zip(gsos, psos, blocktypes):
            if blocktype == 0:
                model = "red_block.egg"
                color = Vec4(1, 0, 0, 1)
            elif blocktype == 1:
                model = "yellow_block.egg"
                color = Vec4(1, 1, 0, 1)

            gso.setColor(color)
            gso.set_model(model)
            self._block_physics(pso)

    def mass_colors(self, sso):
        """Settings for different colored inference mass towers."""
        psos = sso.descendants(type_=RBSO)
        gsos = sso.descendants(type_=GSO)
        blocktypes = get_blocktypes(sso)

        for gso, pso, blocktype in zip(gsos, psos, blocktypes):
            if blocktype == 0:
                color = Vec4(1, 0, 0, 1)
            elif blocktype == 1:
                color = Vec4(1, 1, 0, 1)

            gso.setColor(color)
            gso.set_model("block.egg")
            self._block_physics(pso)

    def apply(self, sso, style):
        style_func = getattr(self, style)
        style_func(sso)

        # for cpo_pth in cpo_pths:
        #     sso = SSO.load_tree(cpo_pth)
        #     style_func(sso)
        #     sso.save_tree(cpo_pth)
