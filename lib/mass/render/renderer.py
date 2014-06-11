# Builtin
import os
import logging
import tempfile
import subprocess as sp
import sys
# External
from path import path
# Panda3D
from libpanda import Vec4
# Scenesim
from scenesim.display.viewer import setup_bullet
from scenesim.objects.gso import GSO
# Local
from viewer import ViewTowers
from mass.render import tasks

logger = logging.getLogger("mass.render")


class RenderMovies(ViewTowers):

    def __init__(self, script, force, fps, ext):
        ViewTowers.__init__(self, script)
        for key in self.permanent_events:
            logger.debug("ignoring key '%s'", key)
            self.ignore(key)
        self.accept("escape", self.cancel)

        self.render_root = None
        self.num_to_do = len(self.options['stimulus'])

        self.fps = fps
        self.ext = ext
        self.encode = " ".join([
            "mencoder",
            "-really-quiet",
            "mf://%(framespth)s\*.%(fmt)s",
            "-mf fps=%(fps)d:type=%(fmt)s",
            "-ovc lavc",
            "-lavcopts vcodec=mpeg4:vbitrate=3200",
            "-oac copy",
            "-o %(moviepth)s"])

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
        if task.getElapsedFrames() > 2:
            return task.done
        return task.cont

    def hide_stimulus(self, task):
        self.sso.detachNode()

    def show_stimulus(self, task):
        self.sso.reparentTo(self.scene)

    def save_screenshot(self, phase, task):
        ext = self.ext
        outdir = self.render_root

        if not outdir.exists():
            logger.debug("Creating output directory %s", outdir)
            outdir.makedirs_p()

        ss_name = "%s~%s.%s" % (self.sso.getName(), phase, ext)
        ss_path = outdir.joinpath(ss_name)
        self.screenshot(
            namePrefix=ss_path,
            defaultFilename=False)

        logger.info("Saved screenshot to %s", ss_path.relpath())

    def start_recording(self, time, task):
        # create a temporary directory for the frame files
        self.render_path = path(tempfile.mkdtemp())
        logger.debug("Temporary render path is %s", self.render_path)
        self.render_prefix = self.render_path.joinpath("movie")
        self.movie(
            namePrefix=self.render_prefix,
            duration=time,
            fps=self.fps,
            format=self.ext,
            sd=4)

    def stop_recording(self, phase, task):
        if self.taskMgr.hasTaskNamed(self.render_prefix + "_task"):
            raise ValueError("Movie task is still running")

        outdir = self.render_root

        # remove the first file, because it is sometimes corrupted
        files = sorted(path(self.render_path).listdir())
        to_remove = files.pop(0)
        logger.debug("Deleting file %s", to_remove)
        os.remove(to_remove)

        # encode the files into a movie
        movie_name = "%s~%s.avi" % (self.sso.getName(), phase)
        movie_path = outdir.joinpath(movie_name)
        cmd = self.encode % {
            'framespth': self.render_prefix,
            'fmt': self.ext,
            'fps': self.fps,
            'moviepth': movie_path
        }
        logger.debug(cmd)
        sp.call(cmd, shell=True)
        logger.info("Saved video to %s", movie_path.relpath())

        # remove the temporary directory and files
        logger.debug("Removing temporary directory %s", self.render_path)
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

    def reset_camera(self):
        self.start_time = self.taskMgr.globalClock.getFrameTime()

    def spin_camera(self, start, rot, time, task):
        end = start + rot
        self.camera_accum = self._get_elapsed()

        if self.camera_accum > time:
            self.camera_rot.setH(end)
            return task.done

        else:
            amt = self.camera_accum / time
            self.camera_rot.setH(start + (amt * (end - start)))
            return task.cont

    def mark_finished(self, i, task):
        script_path = self.options['script_path'][i]
        script_index = self.options['script_index'][i]
        tasks.mark_finished(script_path, script_index)
        self.num_to_do -= 1

    def stop_task(self, task):
        if self.taskMgr.hasTaskNamed(task):
            self.taskMgr.remove(task)

    def cancel(self):
        sys.exit(100)

    def exit(self):
        sys.exit(0)

    def execute_script_step(self, t):
        if len(self.script) == 0:
            return self.next()

        howlong, taskname, args = self.script.pop(0)
        task = getattr(self, taskname)

        argstr = "" if len(args) == 0 else ", ".join(map(str, args))
        longstr = " " if not howlong else " for %.1fs " % howlong
        logger.debug("Executing script step: %s(%s)%s",
                     taskname, argstr, longstr)

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
        elif taskname == "spin_camera":
            self.reset_camera()

    def make_script(self, i):
        show_feedback = self.options['feedback'][i]
        drop_occluder = self.options['occlude'][i]
        full_render = self.options['full_render'][i]
        ptime = self.options['presentation_time'][i] - 1.0
        ftime = self.options['feedback_time'][i]
        camstart = self.options['camera_start'][i]
        camrot = self.options['camera_spin'][i]

        script = []

        def add_step(howlong, task, *args):
            script.append([howlong, task, args])

        # set the camera angle
        add_step(None, "set_camera_angle", camstart)

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
        add_step(None, "spin_camera", camstart, camrot, ptime)

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

        add_step(None, "mark_finished", i)

        return script

    def next(self, steps=1):
        """Go forward one SSO."""
        i = self.ssos.index(self.sso) + steps
        if i == len(self.ssos):
            return self.exit()
        else:
            self.goto_sso(i)

    def goto_sso(self, i):
        ViewTowers.goto_sso(self, i)
        logger.info("%d stimuli left to render", self.num_to_do)

        self.render_root = path(self.options['render_root'][i])
        self.script = self.make_script(i)
        self.taskMgr.add(self.execute_script_step, "execute_script_step")


def render(script, force, fps, ext):
    # setup Bullet physics
    bbase = setup_bullet()

    # create the instance
    app = RenderMovies(script, force, fps, ext)
    app.init_physics(bbase)
    app.init_ssos()
    app.run()
