import sys
import os
import subprocess as sp
import shutil

from optparse import OptionParser

from cogphysics.core import cpObject
from cogphysics.core.graphics import PandaGraphics as Graphics

from scenes import OriginalTowerScene
from scenes import PredictionTowerScene
from scenes import InferenceTowerScene
from view_towers_base import ViewTowers

CPOPATH = "../../stimuli/obj/old"
LISTPATH = "../../stimuli/lists"
LOGPATH = "../../stimuli/meta"
MOVIEPATH = "../../stimuli/render"


class RenderMovies(ViewTowers):

    def __init__(self, scenes, target, basename, scenetype, feedback,
                 occlude, counterbalance):

        # set the scene type class
        if scenetype == "original":
            st = OriginalTowerScene
        elif scenetype == "prediction":
            st = PredictionTowerScene
        elif scenetype == "inference":
            st = InferenceTowerScene
        else:
            raise ValueError("invalid scene type: %s" % scenetype)

        # compute various paths
        self.moviepath = os.path.join(MOVIEPATH, target)
        print "Rendering movies to '%s'" % self.moviepath
        if not os.path.exists(self.moviepath):
            print "Creating movie path because it does not exist"
            os.makedirs(self.moviepath)
        self.logpath = os.path.join(
            LOGPATH, "%s-rendering-info.csv" % target)

        # initialize the viewer app
        RenderMovies.towerscene_type = st
        cpopath = os.path.join(CPOPATH, basename)
        ViewTowers.__init__(self, cpopath, playback=True)

        # save parameters
        self.scenetype = scenetype
        self.f_feedback = feedback
        self.f_occlude = occlude
        self.f_counterbalance = counterbalance
        self.f_render_full = None

        # load/save rendering info
        self.processScenes(scenes)
        self.sidx = -1

        # video rendering settings
        self.currmovie = None
        self.fps = 30
        self.fmt = "png"
        self.factor = 2
        self.fulltime = 6.5
        self.fbtime = 2.5
        self.time = 4.0
        self.encode = " ".join([
            "mencoder",
            "-really-quiet",
            "mf://%s\*.%s",
            "-mf fps=%d:type=%s",
            "-ovc lavc",
            "-lavcopts vcodec=mpeg4:vbitrate=3200",
            "-oac copy",
            "-o %s.avi"])

        # occluder settings
        self.occluder_start_pos = 10
        self.occluder_drop_time = 0.4
        self.occluder_accum = 0
        self.createOccluder()

        # camera settings
        self.total_cam_time = 3.0
        self.cam_accum = 0
        base.camera.setPos(0, -8, 2.75)
        base.camera.lookAt(0, 0, 1.5)

        self.render_stage = None
        self.is_init = True

        # override physics and camera methods
        self.taskMgr.remove("physics")
        self.taskMgr.remove("camera")

        # start rendering!
        self.taskMgr.add(self.renderLoop, "renderLoop")
        self.taskMgr.doMethodLater(0.1, self.initRender, "initRender")

    ##################################################################
    # Initialization
    ##################################################################

    def processScenes(self, scenes):
        # rename stimuli to have a counterbalance parameter, if the
        # counterbalance flag was set
        if self.f_counterbalance:
            self.scenes = []
            for scene in scenes:
                sn = "_" if "~" in scene else "~"
                self.scenes.append(scene + sn + "cb-0")
                self.scenes.append(scene + sn + "cb-1")
        else:
            self.scenes = scenes

        # load old stimlus info
        stiminfo, angles = self.loadRenderingInfo()

        self.stiminfo = stiminfo

    def createOccluder(self):
        """Create the occluder, which drops down after people view the
        tower before they make their response, to ensure everybody
        sees the stimulus for the same amount of time.

        """
        self.occluder = cpObject(
            label='occluder',
            model='cylinderZ',
            shape='cylinderZ',
            scale=(1.5, 1.5, 5),
            pos=(0, 0, self.occluder_start_pos),
            color=(0.2, 0.2, 0.2, 1),
            graphics=Graphics)
        self.occluder.disableGraphics()

    def loadRenderingInfo(self):
        stiminfo = {}
        angles = {}

        # no previous rendering info exists
        if not os.path.exists(self.logpath):
            return stiminfo, angles

        # read csv contents
        with open(self.logpath, "r") as fh:
            lines = [x for x in fh.read().split("\n") if x != '']
        # figure out the keys and which one corresponds to 'stimulus'
        keys = lines[0].split(",")
        sidx = keys.index('stimulus')
        del keys[sidx]
        # build a dictionary of stimuli info
        for line in lines[1:]:
            parts = line.split(",")
            stim = parts[sidx]
            del parts[sidx]
            stiminfo[stim] = dict(zip(keys, parts))

        return stiminfo, angles

    ##################################################################
    # Tasks
    ##################################################################

    def renderLoop(self, task):

        if self.is_init or self.is_capturing:
            return task.cont

        for taskname in ("simulatePhysics", "dropOccluder", "raiseOcccluder"):
            if self.taskMgr.hasTaskNamed(taskname):
                print "Warning: removing task '%s'" % taskname
                self.taskMgr.remove(taskname)

        # render the stimulus presentation
        if self.render_stage is None:
            # capture an image of just the floor
            self.captureImage("%s-floor.png" % self.currmovie)
            # then enable graphics for the tower
            self.towerscene.tower.enableGraphics()

            # start rendering
            self.render_stage = "stimulus"
            self.startRender()
            print " + spin camera"
            self.taskMgr.add(self.spinCamera, "spinCamera")

            print "Rendering stimulus..."
            return task.cont

        # encode the video we (may have) just recorded
        self.encodeVideo()

        # render (separate) feedback
        if self.render_stage == "stimulus" and self.f_feedback:

            self.currscene = self.scenes[self.sidx] + "-feedback"
            self.currmovie = os.path.join(self.moviepath, self.currscene)

            # remove old files
            files = [os.path.join(self.moviepath, f)
                     for f in os.listdir(self.moviepath)]
            files = [f for f in files if f.startswith(self.currmovie)]
            map(os.remove, files)

            # start the recording
            self.render_stage = "feedback"
            self.startRender()

            if self.f_occlude:
                print " + raise occluder"
                self.taskMgr.doMethodLater(
                    0.5, self.raiseOccluder, "raiseOccluder")
            else:
                print " + simulate physics"
                self.taskMgr.doMethodLater(
                    0.5, self.simulatePhysics, "simulatePhysics")

            print "Rendering feedback..."
            return task.cont

        # we're done with all recording for this scene
        elif (self.render_stage == "feedback" or
              (self.render_stage == "stimulus" and
               not self.f_feedback)):

            # flag that we're initializing the render process, so
            # other tasks don't do anything in the meantime
            self.is_init = True
            self.render_stage = None
            self.taskMgr.add(self.initRender, "initRender")

        return task.cont

    def initRender(self, task):

        # check if we're done or not
        if self.sidx >= len(self.scenes) - 1:
            print "\n" + "="*70
            print "Done!"
            print
            sys.exit()

        # load the next scene in the sequence
        print "-"*70
        self.nextScene()
        self.currscene = self.scenes[self.sidx]
        self.currmovie = os.path.join(self.moviepath, self.currscene)
        print "\n" + "="*70
        print "[%d/%d] %s" % (self.sidx+1, len(self.scenes), self.currscene)
        print "-"*70

        # skip this scene, if the files exist
        if self.filesExist(self.currmovie):
            print " ** Files already exist, skipping..."
            return task.cont

        # reset the occluder's position
        pos = self.occluder.pos
        pos[2] = self.occluder_start_pos
        self.occluder.pos = pos
        self.occluder.disableGraphics()

        # set if full recording
        full = self.stiminfo[self.currscene]['full']
        self.f_render_full = True if full == "True" else False

        # set the camera angle
        ang = int(self.stiminfo[self.currscene]['angle'])
        self.rotating.setH(ang)

        # remove old files
        files = [os.path.join(self.moviepath, f)
                 for f in os.listdir(self.moviepath)]
        files = [f for f in files if f.startswith(self.currmovie)]
        map(os.remove, files)

        # set block colors
        color0 = self.stiminfo[self.currscene]['color0']
        color1 = self.stiminfo[self.currscene]['color1']
        if color0 and color1:
            print "Colors are: %s, %s" % (color0, color1)
            self.towerscene.setBlockProperties(
                color0=color0, color1=color1)

        # print stability
        stable = self.stiminfo[self.currscene]['stable']
        if stable == "True":
            print "Tower is STABLE"
        else:
            print "Tower is UNSTABLE"

        # load graphics for the scene
        self.towerscene.setGraphics()
        self.towerscene.tower.disableGraphics()

        # set the init flag back to false so renderLoop will start
        # rendering
        self.is_init = False

        return task.done

    def spinCamera(self, task):
        time = globalClock.getDt()
        self.cam_accum += time
        if self.cam_accum > self.total_cam_time:
            # reset the camera
            self.cam_accum = 0
            ret = task.done
            print " - spin camera"

            # start the next step in the scene
            if self.f_render_full:
                print " + simulate physics"
                self.taskMgr.doMethodLater(
                    1.0, self.simulatePhysics, "simulatePhysics")
            elif self.f_occlude:
                print " + drop occluder"
                self.taskMgr.doMethodLater(
                    0.5, self.dropOccluder, "dropOccluder")
        else:
            ret = task.cont

        deg = time * 60
        self.rotating.setH(self.rotating.getH() + deg)

        return ret

    def dropOccluder(self, task):
        # enable occluder graphics
        self.occluder.enableGraphics()

        # helper variables
        start = self.occluder_start_pos
        end = self.occluder.scale[2] / 2.0
        time = self.occluder_drop_time

        # amount of time that's passed since we started dropping the
        # occluder
        self.occluder_accum += globalClock.getDt()

        if self.occluder_accum > time:
            new_zpos = end
            self.occluder_accum = 0
            ret = task.done
            print " - drop occluder"

        else:
            amt = self.occluder_accum / time
            new_zpos = start + (amt * (end - start))
            ret = task.cont

        # compute the new position of the occluder
        pos = self.occluder.pos
        pos[2] = new_zpos
        self.occluder.pos = pos

        return ret

    def raiseOccluder(self, task):
        # helper variables
        end = self.occluder_start_pos
        start = self.occluder.scale[2] / 2.0
        time = self.occluder_drop_time

        # amount of time that's passed since we started raising the
        # occluder
        self.occluder_accum += globalClock.getDt()

        if self.occluder_accum > time:
            new_zpos = end
            self.occluder_accum = 0
            ret = task.done
            print " - raise occluder"

            # disable occluder graphics
            self.occluder.enableGraphics()
            # enable physics
            print " + simulate physics"
            self.taskMgr.add(self.simulatePhysics, "simulatePhysics")

        else:
            amt = self.occluder_accum / time
            new_zpos = start + (amt * (end - start))
            ret = task.cont

        # compute the new position of the occluder
        pos = self.occluder.pos
        pos[2] = new_zpos
        self.occluder.pos = pos

        return ret

    def simulatePhysics(self, task):
        # don't actually simulate anything if it's stable
        stable = self.stiminfo[self.currscene]['stable']
        if stable:
            print " - simulate physics"
            return task.done
        # seek to new time in the playback
        self.phys_accum += globalClock.getDt() * self.factor
        timedelta = self.towerscene.scene.pbSeekTime(self.phys_accum)
        self.pb_timePlaying += timedelta
        self.phys_accum -= timedelta
        # check if we're past the length of the playback
        if self.pb_timePlaying >= self.pb_maxTimePlaying:
            print " - simulate physics"
            return task.done
        return task.cont

    ##################################################################
    # Helper methods
    ##################################################################

    def encodeVideo(self):
        # find the png files that were generated by the ShowBase
        # video recorder
        files = os.listdir(self.moviepath)
        files = [os.path.join(self.moviepath, f) for f in files]
        files = sorted([
            f for f in files
            if (f.startswith(self.currmovie + "_") and
                f.endswith(".%s" % self.fmt))])
        # remove the first file, because it is sometimes corrupted
        os.remove(files.pop(0))

        # encode the files into a movie
        cmd = self.encode % (
            self.currmovie, self.fmt,
            self.fps, self.fmt, self.currmovie)
        print cmd
        sp.call(cmd, shell=True)

        # save the first and last frame
        shutil.move(files[0], "%s~A.%s" % (
            "_".join(files[0].split("_")[:-1]), self.fmt))
        shutil.move(files[-1], "%s~B.%s" % (
            "_".join(files[-1].split("_")[:-1]), self.fmt))
        # remove everything else
        map(os.remove, files[1:-1])

    def startRender(self):
        # choose duration, depending on which stage we're at
        if self.render_stage == "stimulus":
            if self.f_render_full:
                time = self.fulltime
            else:
                time = self.time
        elif self.render_stage == "feedback":
            time = self.fbtime
        else:
            raise ValueError("invalid render stage: %s" % self.render_stage)

        # start rendering
        self.movie(namePrefix=self.currmovie, duration=time,
                   fps=self.fps, format=self.fmt, sd=4)

    @property
    def is_capturing(self):
        if self.taskMgr.hasTaskNamed(self.currmovie + "_task"):
            is_capturing = True
        else:
            is_capturing = False
        return is_capturing

    def filesExist(self, basename):
        scenename = self.scenes[self.sidx]
        suffixes = [
            ".avi",
            "~A.%s" % self.fmt,
            "~A.%s" % self.fmt,
            "-floor.%s" % self.fmt
            ]

        if self.f_feedback:
            suffixes.extend([
                "-feedback.avi",
                "-feedback~A.%s" % self.fmt,
                "-feedback~B.%s" % self.fmt,
                ])
        for suffix in suffixes:
            filename = basename + suffix
            if not os.path.exists(filename):
                return False
        return True


if __name__ == "__main__":
    usage = "usage: %prog [options] target list1 [list2 ... listN]"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "-s", "--stype", dest="stype", action="store",
        help="stimulus type, e.g. mass-learning [required]",
        metavar="STIM_TYPE")
    parser.add_option(
        "-o", "--original", dest="scenetype",
        action="store_const", const="original",
        help="use original stability (non-mass) towers")
    parser.add_option(
        "-p", "--prediction", dest="scenetype",
        action="store_const", const="prediction",
        help="use mass prediction (green/gray) towers")
    parser.add_option(
        "-i", "--inference", dest="scenetype",
        action="store_const", const="inference",
        help="use mass inference (red/yellow) towers")
    parser.add_option(
        "-c", "--counterbalance",
        action="store_true", dest="counterbalance", default=False,
        help="counterbalance stimuli colors")
    parser.add_option(
        "--feedback",
        action="store_true", dest="feedback",
        help="render feedback")
    parser.add_option(
        "--no-feedback",
        action="store_false", dest="feedback",
        help="do not render feedback")
    parser.add_option(
        "--occlude",
        action="store_true", dest="occlude", default=False,
        help="drop occluder after stimulus presentation")

    (options, args) = parser.parse_args()
    if len(args) == 0:
        raise ValueError("no target directory name specified")
    elif len(args) == 1:
        print("Warning: no stimuli lists passed. "
              "Assuming '%s' instead." % options.stype)
        lists = [options.stype]
        target = args[0]
    else:
        lists = args[1:]
        target = args[0]

    scenes = []
    for listname in lists:
        listpath = os.path.join(LISTPATH, listname)
        with open(listpath, "r") as fh:
            cpos = [x for x in fh.read().strip().split("\n") if x != ""]
            scenes.extend(cpos)

    feedback = options.feedback
    scenetype = options.scenetype
    occlude = options.occlude
    counterbalance = options.counterbalance

    app = RenderMovies(
        scenes, target, options.stype, scenetype, feedback,
        occlude, counterbalance)
    app.run()
