# Builtin
import os
import warnings
import logging
from argparse import ArgumentParser
from copy import deepcopy
# External
import numpy as np
from path import path
# Panda3D
import pandac.PandaModules as pm
import panda3d.core as p3d
from libpanda import TextureStage
# Scenesim
from scenesim.display.viewer import Viewer
from scenesim.display.viewer import load as load_ssos
from scenesim.display.viewer import setup_bullet
from scenesim.objects.gso import GSO
from scenesim.objects.pso import RBSO

logging.basicConfig(level=logging.INFO)

# load panda configuration
p3d.loadPrcFile("../config/Config.prc")
CPO_PATH = path(p3d.ConfigVariableString("cpo-path", "").get_value())

STIMTYPES = {
    'mass-example': "mass_red_yellow",
    'mass-learning': "mass_red_yellow",
    'mass-learning-training': "original",

    'mass-oneshot-F': "mass_colors",
    'mass-oneshot-example-F': "mass_colors",
    'mass-oneshot-training-F': "original",
    'mass-inference': "mass_colors",

    'mass-prediction-stability': "mass_plastic_stone",
    'mass-prediction-direction': "mass_plastic_stone",

    'stability-example-stable': "original",
    'stability-example-stable-F': "original",
    'stability-example-unstable': "original",
    'stability-example-unstable-F': "original",
    'stability-original': "original",
    'stability-sameheight': "original",
    'stability-unstable': "original",
}


class ViewTowers(Viewer):

    def __init__(self):

        self.accept("c", self.flip_colors)

        Viewer.__init__(self)

        self.place_camera()
        self.create_lights()
        self.win.setClearColor((0.25, 0.25, 0.45, 1.0))
        self.disableMouse()

        self.stimtype = None
        self.flipped = False
        self._blocktype_cache = None

    def place_camera(self):
        self.cameras.setPos(0, -8, 2.75)
        self.look_at.setPos(0, 0, 1.5)
        self.cameras.lookAt(self.look_at)

    def create_lights(self):
        angle1 = -155
        angle2 = 180 + angle1
        r1 = r2 = 5
        z = 8
        at = (1, 0, .02)
        color = tuple(np.ones(4)*3)
        slcolor = 1

        p1 = (np.cos(np.radians(angle1))*r1, np.sin(np.radians(angle1))*r1, z)
        p2 = (np.cos(np.radians(angle2))*r2, np.sin(np.radians(angle2))*r2, z)

        # Create point lights
        plight1 = pm.PointLight('plight1')
        plight1.setColor(color)
        plight1.setAttenuation(at)
        plnp1 = self.lights.attachNewNode(plight1)
        plnp1.setPos(p1)
        plnp1.lookAt(0, 0, 0)
        self.render.setLight(plnp1)

        plight2 = pm.PointLight('plight2')
        plight2.setColor(tuple(np.array(color)-slcolor))
        plight2.setAttenuation(at)
        plnp2 = self.lights.attachNewNode(plight2)
        plnp2.setPos(p2)
        plnp2.lookAt(0, 0, 0)
        self.render.setLight(plnp2)

        # Create ambient light
        alnp = self.lights.find('alight')
        alnp.node().setColor((0.2, 0.2, 0.2, 1.0))

        # Create spotlight
        slnp = self.lights.find('slight')
        slnp.node().setColor((slcolor, slcolor, slcolor, 1))
        slnp.node().setAttenuation(at)
        slnp.setPos(p2)
        slnp.lookAt(0, 0, 0)

    def flip_colors(self, flip=None, update=True):
        if self.stimtype in "original":
            if flip:
                warnings.warn("Cannot flip colors for original towers")
            return

        if flip is None:
            self.flipped = not(self.flipped)
        else:
            self.flipped = bool(flip)
        logging.info("self.flipped is now %s" % self.flipped)

        if update:
            self._apply_blocktype_props()

    @property
    def curr_blocktypes(self):
        return [int(x) for x in self.sso.getName().split("_")[-1]]

    @property
    def curr_gsos(self):
        return self.sso.descendants(type_=GSO, names="block")

    @property
    def curr_psos(self):
        return self.sso.descendants(type_=RBSO, names="block")

    def _store_blocktype_props(self):
        props = {}
        for gso, blocktype in zip(self.curr_gsos, self.curr_blocktypes):
            color = gso.getColor()
            model = gso.get_model()
            if blocktype not in props:
                props[blocktype] = []
            props[blocktype].append({'color': color, 'model': model})
        self._blocktype_cache = props

    def _apply_blocktype_props(self, bitstr=None):
        if bitstr is None:
            if self.flipped:
                bitstr = [1-x for x in self.curr_blocktypes]
            else:
                bitstr = self.curr_blocktypes

        self.sso.destroy_tree(tags=("model",))
        props = deepcopy(self._blocktype_cache)
        for gso, blocktype in zip(self.curr_gsos, bitstr):
            prop = props[blocktype].pop(0)
            gso.apply_prop(prop)
        self.sso.init_tree(tags=("model",))
        map(self._set_block_graphics, self.curr_gsos, bitstr)

    def _restore_blocktype_props(self):
        self._apply_blocktype_props(self.curr_blocktypes)

    def _set_block_graphics(self, block, blocktype):
        # use a random seed based on the block's name, so we can have
        # "random" properties that are actually always the same for
        # any given block
        rso = np.random.RandomState(abs(hash(repr(block))))

        # original towers
        if self.stimtype == "original":
            block.clearMaterial()
            block.clearTexture()
            tex = self.loader.loadTexture('noise.rgb')
            scl = np.max(block.getScale()) / 5
            offset = (
                rso.rand()*scl*2 - scl,
                rso.rand()*scl*2 - scl)
            scale = (2.0, 2.0)

            ts = TextureStage('ts_%s' % block.getName())
            block.setTexture(ts, tex, 1)
            block.setTexOffset(ts, *offset)
            block.setTexScale(ts, *scale)

        # green plastic/gray stone mass towers
        elif self.stimtype == "mass_plastic_stone":
            hpr = rso.rand(3)
            offset = rso.rand(2)
            scale = rso.rand(2)

            if blocktype == 0:
                tex = self.loader.loadTexture('stripes.png')
                scl = np.max(block.getScale()) / 5
                hpr *= 180
                offset = offset*scl*2 - scl
                scale = (2.0, 2.0)

            elif blocktype == 1:
                tex = self.loader.loadTexture('granite-grayscale.jpg')
                scl = np.max(block.getScale()) / 10
                hpr *= 20
                offset = offset*scl*2 - scl
                scale = scale*1.5 + 0.5

            ts = TextureStage('ts_%s' % block.getName())
            block.setTexture(ts, tex, 1)
            block.setTexHpr(ts, *hpr)
            block.setTexOffset(ts, *offset)
            block.setTexScale(ts, *scale)

        # red/yellow mass towers
        elif self.stimtype == "mass_red_yellow":
            pass

        # arbitrary color mass towers
        elif self.stimtype == "mass_colors":
            block.clearMaterial()
            block.clearTexture()
            tex = self.loader.loadTexture('noisy-grayscale.jpg')
            scl = np.max(block.getScale()) / 10
            offset = rso.rand(2)*scl*2 - scl
            scale = (0.2, 0.2)

            ts = TextureStage('ts_%s' % block.getName())
            block.setTexture(ts, tex, 1)
            block.setTexOffset(ts, *offset)
            block.setTexScale(ts, *scale)

    def _set_block_physics(self, block, blocktype):
        # original towers
        if self.stimtype == "original":
            pass

        # mass towers
        elif self.stimtype.startswith("mass"):
            if blocktype == 1:
                mass = block.get_mass()
                mass *= (10 ** self.kappa)
                block.set_mass(mass)

    def init_ssos(self, opts):
        """ Initialize the ssos."""
        self.options = deepcopy(opts)

        # load the actual sso objects from disk and do the default
        # Viewer initialization
        ssos = load_ssos(opts['stimulus'])
        Viewer.init_ssos(self, ssos)

        # initialize the floor sso
        floor_path = os.path.join(
            CPO_PATH, "floors", "round-wooden-floor.cpo")
        self.floor = load_ssos([floor_path])[0]
        self.floor.setPos((0, 0, -0.5))
        self.floor.reparentTo(self.scene)
        self.floor.init_tree(tags=("model", "shape"))

    def goto_sso(self, i):
        if self._blocktype_cache:
            self._restore_blocktype_props()
            self._blocktype_cache = None

        self.stimtype = self.options['stimtype'][i]
        self.kappa = self.options['kappa'][i]

        Viewer.goto_sso(self, i)

        self._store_blocktype_props()
        self.flip_colors(flip=self.options['flip_colors'][i], update=False)
        self._apply_blocktype_props()

    def attach_physics(self):
        map(self._set_block_physics, self.curr_psos, self.curr_blocktypes)
        Viewer.attach_physics(self)


def parseargs():

    parser = ArgumentParser()
    parser.add_argument(
        "stimuli", metavar="stim", type=str, nargs="+",
        help="path to stimulus")
    parser.add_argument(
        "-s", "--stype", dest="stype", action="store",
        help=("stimulus type. If not provided, it will be inferred "
              "from the paths to the stimuli."),
        choices=sorted(set(STIMTYPES.values())))
    parser.add_argument(
        "-k", "--kappa", dest="kappa",
        action="store", type=float, default=0.0,
        help="log10 mass ratio (only for mass towers)")
    parser.add_argument(
        "--flip-colors",
        action="store_true", dest="flip", default=False,
        help="swap block colors")
    parser.add_argument(
        "--angle",
        action="store", dest="angle", type=int,
        help="initial camera angle")

    args = parser.parse_args()
    stims = [path(x).abspath() for x in args.stimuli]
    if args.stype:
        stimtypes = [args.stype]*len(stims)
    else:
        stimtypes = [STIMTYPES[x.splitall()[-2]] for x in stims]

    N = len(stims)
    opts = {
        'stimulus': stims,
        'stimtype': stimtypes,
        'kappa': [args.kappa]*N,
        'flip_colors': [args.flip]*N,
        'angle': [args.angle]*N
    }
    return opts


if __name__ == "__main__":

    # command line arguments
    opts = parseargs()
    # setup Bullet physics
    bbase = setup_bullet()

    # create the instance
    app = ViewTowers()
    app.init_physics(bbase)
    app.init_ssos(opts)
    app.run()
