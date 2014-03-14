# Builtin
import os
import logging
from copy import deepcopy
# External
import numpy as np
# Panda3D
import pandac.PandaModules as pm
import panda3d.core as p3d
# Scenesim
from scenesim.display.viewer import Viewer
from scenesim.display.viewer import load as load_ssos
from scenesim.display.viewer import setup_bullet
from scenesim.objects.gso import GSO
from scenesim.objects.pso import RBSO

from mass import CPO_PATH
from mass.stimuli import GSOStyler, PSOStyler
from mass.stimuli.util import get_blocktypes

logger = logging.getLogger("mass.render")


class ViewTowers(Viewer):

    def __init__(self, options):

        Viewer.__init__(self)

        self.options = deepcopy(options)

        self.place_camera()
        self.create_lights()
        self.win.setClearColor((0.05, 0.05, 0.1, 1.0))
        self.disableMouse()

        self.stimtype = None
        self._blocktype_cache = None

    def place_camera(self):
        self.cameras.setPos(0, -8, 2.75)
        self.look_at.setPos(0, 0, 1.5)
        self.cameras.lookAt(self.look_at)

    def create_lights(self):
        # function for converting cylindrical coordinates to cartesian
        # coordinates
        rtz2xyz = lambda r, t, z: (r*np.cos(t), r*np.sin(t), z)

        # positions for point lights
        plight_pos = [
            rtz2xyz(1.5, 4*np.pi/12., 0),
            rtz2xyz(1.5, 12*np.pi/12., 0),
            rtz2xyz(1.5, 20*np.pi/12., 0),
            (0, 0, 1.3),
        ]

        # create point lights
        self.plights = p3d.NodePath("plights")
        for i, pos in enumerate(plight_pos):
            plight = pm.PointLight('plight%d' % i)
            plight.setColor((0.5, 0.5, 0.5, 1.0))
            plight.setAttenuation((0, 0, 0.5))
            plnp = self.plights.attachNewNode(plight)
            plnp.setPos(pos)
            self.render.setLight(plnp)
        self.plights.reparentTo(self.lights)

        # update the position and color of the spotlight
        slnp = self.lights.find('slight')
        slnp.setPos((8, 6, 20))
        slnp.lookAt(self.look_at)
        slnp.node().setColor((1, 1, 1, 1))

        # update the color of the ambient light
        alnp = self.lights.find('alight')
        alnp.node().setColor((0.2, 0.2, 0.2, 1))

    @property
    def curr_blocktypes(self):
        return get_blocktypes(self.sso)

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

    def _apply_blocktype_props(self):
        bitstr = self.curr_blocktypes
        self.sso.destroy_tree(tags=("model",))
        props = deepcopy(self._blocktype_cache)
        for gso, blocktype in zip(self.curr_gsos, bitstr):
            prop = props[blocktype].pop(0)
            gso.apply_prop(prop)
        self.sso.init_tree(tags=("model",))
        styler = GSOStyler(self.loader)
        for i, gso in enumerate(self.curr_gsos):
            styler.apply(
                gso, self.stimtype,
                blocktype=bitstr[i],
                color0=self.color0,
                color1=self.color1)

    def init_ssos(self):
        """ Initialize the ssos."""

        # load the actual sso objects from disk and do the default
        # Viewer initialization
        ssos = load_ssos(self.options['stimulus'])
        Viewer.init_ssos(self, ssos)

        # initialize the floor sso
        floor_path = os.path.join(
            CPO_PATH, "floors", "round-wooden-floor.cpo")
        self.floor = load_ssos([floor_path])[0]
        gso, = self.floor.descendants(type_=GSO)
        PSOStyler().apply(self.floor, "floor")
        GSOStyler(self.loader).apply(gso, "floor")
        self.floor.reparentTo(self.scene)
        self.floor.init_tree(tags=("model", "shape"))

        # give it a little extra ambient light
        alight = pm.AmbientLight('alight2')
        alight.setColor((0.6, 0.6, 0.6, 1.0))
        alnp = self.lights.attachNewNode(alight)
        self.floor.setLight(alnp)

    def optimize_camera(self):
        pass

    def goto_sso(self, i):
        if self._blocktype_cache:
            self._blocktype_cache = None

        self.stimtype = self.options['stimtype'][i]
        self.kappa = self.options['kappa'][i]
        self.color0 = self.options['color0'][i]
        self.color1 = self.options['color1'][i]

        Viewer.goto_sso(self, i)

        self._store_blocktype_props()
        self._apply_blocktype_props()

        minb, maxb = self.sso.getTightBounds()
        height = max(2, maxb[2])
        self.plights.setPos(0, 0, height*2/3.)

        logger.info("Showing sso '%s'", self.sso.getName())

    def attach_physics(self):
        styler = PSOStyler()
        for pso, blocktype in zip(self.curr_psos, self.curr_blocktypes):
            styler.apply(
                pso, self.stimtype,
                blocktype=blocktype,
                kappa=self.kappa)
        Viewer.attach_physics(self)

    def physics(self, task):
        """Task: simulate physics."""
        # Elapsed time.
        dt = self._get_elapsed() - self.old_elapsed
        # Update amount of time simulated so far.
        self.old_elapsed += dt
        # Step the physics dt time.
        size_sub = 1. / 1000.
        n_subs = int(dt / size_sub)
        self.bbase.step(dt, n_subs, size_sub)
        return task.cont


def init(options):
    # setup Bullet physics
    bbase = setup_bullet()

    # create the instance
    app = ViewTowers(options)
    app.init_physics(bbase)
    app.init_ssos()
    app.run()
