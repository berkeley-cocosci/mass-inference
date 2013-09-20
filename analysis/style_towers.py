import colorsys

import numpy as np

import pandac.PandaModules
from libpanda import Vec4

from scenesim.objects.sso import SSO
from scenesim.objects.gso import GSO
from scenesim.objects.pso import RBSO


def _block_physics(pso):
    """Physics settings for tower blocks."""
    density = 170.0
    volume = np.prod(pso.getScale())
    mass = density * volume
    pso.set_mass(mass)
    pso.set_friction(0.8 ** 0.5)
    pso.set_restitution(0)


def floor(sso, rso):
    """Settings for round wooden floor SSO."""
    gsos = sso.descendants(type_=GSO)
    assert len(gsos) == 1
    gso = gsos[0]
    gso.set_model("wood_floor.egg")
    gso.setColor((0.45, 0.3, 0.1, 1.0))
    sso.set_friction(0.8 ** 0.5)
    sso.set_restitution(0)
    sso.setScale((10, 10, 1))


def original(sso, rso):
    """Settings for original-type towers (random colors)."""
    psos = sso.descendants(type_=RBSO)
    gsos = sso.descendants(type_=GSO)
    for pso, gso in zip(psos, gsos):
        r, g, b = colorsys.hsv_to_rgb(rso.rand(), 1, 1)
        color = Vec4(r, g, b, 1)

        gso.setColor(color)
        gso.set_model("block.egg")
        _block_physics(pso)


def mass_plastic_stone(sso, rso):
    """Settings for prediction mass towers (plastic/stone blocks)."""
    lcscale = 0.15
    hcscale = 0.1
    clip = lambda x: np.clip(x, 0, 1)

    psos = sso.descendants(type_=RBSO)
    gsos = sso.descendants(type_=GSO)
    bitstr = [int(x) for x in sso.getName().split("_")[-1]]

    for gso, pso, blocktype in zip(gsos, psos, bitstr):
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
        _block_physics(pso)


def mass_red_yellow(sso, rso):
    """Settings for RY inference mass towers (red/yellow blocks)."""
    psos = sso.descendants(type_=RBSO)
    gsos = sso.descendants(type_=GSO)
    bitstr = [int(x) for x in sso.getName().split("_")[-1]]

    for gso, pso, blocktype in zip(gsos, psos, bitstr):
        if blocktype == 0:
            model = "red_block.egg"
            color = Vec4(1, 0, 0, 1)
        elif blocktype == 1:
            model = "yellow_block.egg"
            color = Vec4(1, 1, 0, 1)

        gso.setColor(color)
        gso.set_model(model)
        _block_physics(pso)


def mass_colors(sso, rso):
    """Settings for different colored inference mass towers."""
    psos = sso.descendants(type_=RBSO)
    gsos = sso.descendants(type_=GSO)
    bitstr = [int(x) for x in sso.getName().split("_")[-1]]

    for gso, pso, blocktype in zip(gsos, psos, bitstr):
        if blocktype == 0:
            color = Vec4(1, 0, 0, 1)
        elif blocktype == 1:
            color = Vec4(1, 1, 0, 1)

        gso.setColor(color)
        gso.set_model("block.egg")
        _block_physics(pso)


def apply_style(cpo_pths, style, rso):
    style_func = globals()[style]

    for cpo_pth in cpo_pths:
        sso = SSO.load_tree(cpo_pth)
        style_func(sso, rso)
        sso.save_tree(cpo_pth)
