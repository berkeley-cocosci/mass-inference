import warnings
import colorsys
import json
from mass import CPO_PATH


def get_rgb(color, rso):
    """Add a little bit of variation to the saturation and
    value channels"""
    r = int(color[1:3], base=16) / 255.
    g = int(color[3:5], base=16) / 255.
    b = int(color[5:7], base=16) / 255.
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = max(0, min(1, s + rso.randn() / 8.))
    v = max(0, min(1, v + rso.randn() / 8.))
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    rgba = (r, g, b, 1)
    return rgba


def get_blocktypes(sso):
    types = [int(x) for x in sso.getName().split("_")[-1]]
    return types


def get_style(stimtype):
    stylepath = CPO_PATH.joinpath("styles.json")
    with open(stylepath, "r") as fh:
        styles = json.load(fh)
    return styles[stimtype]


def save_style(style, stimtype):
    stylepath = CPO_PATH.joinpath("styles.json")
    with open(stylepath, "r") as fh:
        styles = json.load(fh)
    if style in styles:
        warnings.warn("overwriting style for stimtype '%s'" % stimtype)
    styles[stimtype] = style
    with open(stylepath, "w") as fh:
        json.dump(styles, fh)
