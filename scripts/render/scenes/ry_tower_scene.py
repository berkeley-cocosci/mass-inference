from cogphysics.core.graphics import PandaGraphics as Graphics
from tower_scene_base import TowerScene


class RYTowerScene(TowerScene):

    def setBlockProperties(self, kappa=None, mu=None, counterbalance=None):
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

        for mat in self.table.graphics.node.findAllMaterials():
            mat.setShininess(0)
            mat.clearSpecular()
            mat.clearAmbient()
            mat.clearDiffuse()
