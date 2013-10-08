# Builtin
import os
import warnings
import logging
import tempfile
import subprocess as sp
from argparse import ArgumentParser
# External
import pandas as pd
from path import path
# Panda3D
import pandac.PandaModules as pm
from libpanda import Vec4
# Scenesim
from scenesim.display.viewer import setup_bullet
from scenesim.objects.gso import GSO
# Local
from view_towers import ViewTowers


class RenderMovies(ViewTowers):

    def __init__(self):
        ViewTowers.__init__(self)
        for key in self.permanent_events:
            if key != "escape":
                logging.debug("ignoring key '%s'" % key)
                self.ignore(key)

        self.encode = " ".join([
            "mencoder",
            "-really-quiet",
            "mf://%(framespth)s\*.%(fmt)s",
            "-mf fps=%(fps)d:type=%(fmt)s",
            "-ovc lavc",
            "-lavcopts vcodec=mpeg4:vbitrate=3200",
            "-oac copy",
            "-o %(moviepth)s.avi"])

        self.occluder_start_pos = 10
        self.occluder_accum = 0
        self.create_occluder()

    def create_occluder(self):
        """Create the occluder, which drops down after people view the
        tower before they make their response, to ensure everybody
        sees the stimulus for the same amount of time.

        """
        self.occluder = GSO('occluder')
        self.occluder.set_model('cylinderZ')
        self.occluder.setScale(1.5, 1.5, 5)
        self.occluder.setPos(0, 0, self.occluder_start_pos)
        self.occluder.setColor(Vec4(0.2, 0.2, 0.2, 1))
        self.occluder.reparentTo(self.scene)
        self.occluder.init_tree(tags=('model',))

    def set_camera_angle(self, angle, task):
        self.camera_rot.setH(angle)

    def thunk(self, task):
        if task.getElapsedFrames() > 1:
            return task.done
        return task.cont

    def hide_stimulus(self, task):
        self.sso.detachNode()

    def show_stimulus(self, task):
        self.sso.reparentTo(self.scene)

    def save_screenshot(self, phase, task):
        ext = self.options['ext']
        outdir = path(self.options['outdir'])

        if not outdir.exists():
            logging.info("Creating output directory %s" % outdir)
            outdir.mkdir_p()

        ss_name = "%s~%s.%s" % (self.sso.getName(), phase, ext)
        ss_path = os.path.join(outdir, ss_name)
        self.screenshot(
            namePrefix=ss_path,
            defaultFilename=False)

        logging.info("Saved screenshot to %s" % ss_path)

    def start_recording(self, time, task):
        ext = self.options['ext']
        fps = self.options['fps']

        # create a temporary directory for the frame files
        self.render_path = tempfile.mkdtemp()
        logging.info("Temporary render path is %s" % self.render_path)
        self.render_prefix = os.path.join(self.render_path, "movie")
        self.movie(
            namePrefix=self.render_prefix,
            duration=time,
            fps=fps,
            format=ext,
            sd=4)

    def stop_recording(self, phase, task):
        if self.taskMgr.hasTaskNamed(self.render_prefix + "_task"):
            raise ValueError("Movie task is still running")

        ext = self.options['ext']
        outdir = self.options['outdir']
        fps = self.options['fps']

        # remove the first file, because it is sometimes corrupted
        files = sorted(path(self.render_path).listdir())
        to_remove = files.pop(0)
        logging.info("Deleting file %s" % to_remove)
        os.remove(to_remove)

        # encode the files into a movie
        movie_name = "%s~%s" % (self.sso.getName(), phase)
        cmd = self.encode % {
            'framespth': self.render_prefix,
            'fmt': ext,
            'fps': fps,
            'moviepth': os.path.join(outdir, movie_name),
        }
        logging.info(cmd)
        sp.call(cmd, shell=True)

        # remove the temporary directory and files
        logging.info("Removing temporary directory %s" % self.render_path)
        path(self.render_path).rmtree()

    def reset_occluder(self):
        self.start_time = self.taskMgr.globalClock.getFrameTime()

    def move_occluder(self, action, time, task):
        # helper variables
        if action == "drop":
            start = self.occluder_start_pos
            end = self.occluder.getScale()[2] / 2.0
        elif action == "raise":
            start = self.occluder.getScale()[2] / 2.0
            end = self.occluder_start_pos
        else:
            raise ValueError("invalid action: %s" % action)

        # amount of time that's passed since we started dropping the
        # occluder
        self.occluder_accum = self._get_elapsed()

        if self.occluder_accum > time:
            new_zpos = end
        else:
            amt = self.occluder_accum / time
            new_zpos = start + (amt * (end - start))

        # compute the new position of the occluder
        pos = self.occluder.getPos()
        pos[2] = new_zpos
        self.occluder.setPos(pos)

        return task.cont

    def wait(self, task):
        return task.cont

    def stop_task(self, task):
        if self.taskMgr.hasTaskNamed(task):
            self.taskMgr.remove(task)

    def execute_script_step(self, t):
        if len(self.script) == 0:
            if self.ssos.index(self.sso) == (len(self.ssos) - 1):
                return self.exit()
            else:
                return self.next()

        howlong, taskname, args = self.script.pop(0)
        task = getattr(self, taskname)

        argstr = "" if len(args) == 0 else ", ".join(map(str, args))
        longstr = " " if not howlong else " for %.1fs " % howlong
        logging.info("Executing script step: %s(%s)%s" %
                     (taskname, argstr, longstr))

        self.taskMgr.add(
            task, taskname,
            extraArgs=args,
            uponDeath=self.execute_script_step,
            appendTask=True)

        if howlong:
            self.taskMgr.doMethodLater(
                howlong, self.stop_task, "remove_task",
                extraArgs=[taskname])

        if taskname == "physics":
            self.reset_physics()
        elif taskname == "move_occluder":
            self.reset_occluder()

    def make_script(self, i):
        show_feedback = self.options['feedback'][i]
        drop_occluder = self.options['occlude'][i]
        full_render = self.options['full_render'][i]
        ptime = self.options['presentation_time'][i] - 1.0
        ftime = self.options['feedback_time'][i]
        angle = self.options['angle'][i]

        script = []

        def add_step(howlong, task, *args):
            script.append([howlong, task, args])

        # set the camera angle
        add_step(None, "set_camera_angle", angle)

        # save a screenshot of the floor -- we need to add these
        # "thunk" steps to render a few frames before we actually take
        # the screenshot, to ensure that the graphics buffer is
        # cleaned out
        add_step(None, "hide_stimulus")
        add_step(None, "thunk")
        add_step(None, "save_screenshot", "floor")
        add_step(None, "show_stimulus")
        add_step(None, "thunk")

        # start recording
        if full_render and show_feedback:
            if drop_occluder:
                rtime = 2.5 + ptime + ftime
            else:
                rtime = 1.5 + ptime + ftime

        else:
            if drop_occluder:
                rtime = 1.5 + ptime
            else:
                rtime = 1.0 + ptime

        add_step(None, "save_screenshot", "stimulus~A")
        add_step(None, "start_recording", rtime)

        # spin the camera
        add_step(ptime, "rotate")

        # then pause for a moment, possibly dropping the occluder
        add_step(1.0, "wait")
        if drop_occluder:
            add_step(0.3, "move_occluder", "drop", 0.3)
            add_step(0.2, "wait")

        # save the stimulus presentation
        if not full_render:
            add_step(None, "stop_recording", "stimulus")
            add_step(None, "save_screenshot", "stimulus~B")

        if show_feedback:
            if not full_render:
                if drop_occluder:
                    rtime = 1.0 + ftime
                else:
                    rtime = 0.5 + ftime
                add_step(None, "save_screenshot", "feedback~A")
                add_step(None, "start_recording", rtime)

            # raise the occluder
            add_step(0.5, "wait")
            if drop_occluder:
                add_step(0.3, "move_occluder", "raise", 0.3)
                add_step(0.2, "wait")

            # simulate physics
            add_step(ftime, "physics")

            # save the feedback
            if full_render:
                add_step(None, "stop_recording", "stimulus")
                add_step(None, "save_screenshot", "stimulus~B")
            else:
                add_step(None, "stop_recording", "feedback")
                add_step(None, "save_screenshot", "feedback~B")

        return script

    def goto_sso(self, i):
        ViewTowers.goto_sso(self, i)
        self.script = self.make_script(i)
        self.taskMgr.add(self.execute_script_step, "execute_script_step")


def parseargs():

    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", dest="config", action="store", type=str,
        required=True, help="path to configuration file")
    parser.add_argument(
        "-d", "--dest", dest="outdir", action="store", type=str,
        required=True, help="directory to save render files")

    encopt = parser.add_argument_group(title="Encoding options")
    encopt.add_argument(
        "--fps",
        action="store", dest="fps", type=float, default=30,
        help="frames per second (default: 30)")
    encopt.add_argument(
        "--extension",
        action="store", dest="ext", type=str, default="png",
        choices=["png", "jpeg"],
        help="file format to save frames as (default: png)")

    args = parser.parse_args()
    config = pd.read_csv(args.config).to_dict('list')
    config.update({
        'outdir': args.outdir,
        'fps': args.fps,
        'ext': args.ext
    })

    return config


if __name__ == "__main__":

    # command line arguments
    config = parseargs()
    # setup Bullet physics
    bbase = setup_bullet()

    # create the instance
    app = RenderMovies()
    app.init_physics(bbase)
    app.init_ssos(config)
    app.run()
