import os
import sys

from optparse import OptionParser

from view_towers import ViewTowers
from scenes import InferenceTowerScene


class ViewInferenceTowers(ViewTowers):

    towerscene_type = InferenceTowerScene

    def toggleWireframe(self):

        if not self.fwireframe:
            # turn on wireframe mode
            self.render.setRenderModeWireframe()
            self.render.setRenderMode(self.render.getRenderMode(), 5.0)

            # remove the table and the environment background from the scene
            self.table.disableGraphics()
            self.environment.hide()

            # turn off shaders
            render.setShaderOff()
            render.setLightOff()

            # set block wireframe colors
            for block in self.towerscene.blocks:
                block.meta['color'] = block.color

        else:
            # turn off wireframe mode
            self.render.setRenderModeFilled()
            self.render.setRenderMode(self.render.getRenderMode(), 1.0)

            # add the table and environment background back in
            self.table.enableGraphics()
            self.environment.show()

            # turn shading back on
            render.setShaderAuto()

            # turn lights back on
            for light in self.lights.getChildren():
                render.setLight(light)

            # set the block colors back to normal
            for idx, block in enumerate(self.towerscene.blocks):
                block.color = block.meta['color']

        # toggle texturing
        self.toggleTexture()

        # our own flag to let us know if wirefram is on or off
        self.fwireframe = not(self.fwireframe)

if __name__ == "__main__":
    usage = ("usage: %prog [options] stimlist"
             "       %prog [options] stim1 ... stimN")
    parser = OptionParser(usage=usage)
    parser.add_option(
        "-s", "--stype", dest="stype",
        help="stimulus type, e.g. mass-learning [required]",
        metavar="STIM_TYPE")
    parser.add_option(
        "-c", "--camstart", dest="cam_start",
        action="store", type="int", default=-10,
        help="initial camera angle",
        metavar="ANGLE")

    (options, args) = parser.parse_args()

    if options.stype is None:
        print "No stimulus type provided, exiting."
        sys.exit(1)

    cpopath = os.path.join("../../stimuli/obj/old", options.stype)
    listpath = "../../stimuli/lists"

    if len(args) > 1:
        scenes = args
    else:
        lp = os.path.join(listpath, args[0])
        if not os.path.exists(lp):
            scenes = args
        else:
            with open(lp, "r") as fh:
                scenes = fh.read().split("\n")
            scenes = [x for x in scenes if x != '']

    app = ViewInferenceTowers(cpopath)
    app.scenes = scenes

    app.cam_start = options.cam_start
    app.rotating.setH(app.cam_start)
    app.sidx = -1
    app.nextScene()
    app.run()
