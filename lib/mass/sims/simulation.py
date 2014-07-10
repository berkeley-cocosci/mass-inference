from contextlib import contextmanager
from datetime import datetime, timedelta
from libpanda import Point3, BitMask32
from mass.stimuli import PSOStyler, get_blocktypes, get_style
from multiprocessing import Process
from pandac.PandaModules import NodePathCollection
from path import path
from scenesim.objects.pso import PSO
from scenesim.objects.sso import SSO
from scenesim.physics.bulletbase import BulletBase
from utils import load_cpo
import multiprocessing as mp
import numpy as np
import sys


def get_force(angle, mag):
    """ Input force angle and magnitude, output force vec and pos."""
    c = 3.
    vec = np.array((-np.cos(angle) / c, -np.sin(angle) / c, 1.)) * mag
    pos = np.array((np.cos(angle), np.sin(angle), -1.0))
    return vec, pos


def read(data, pcpos):
    """Records states of all pcpos."""
    data[:] = [np.hstack((pcpo.getPos(), pcpo.getQuat())) for pcpo in pcpos]


class BaseSimulationError(Exception):
    """Base class for simulation exceptions."""
    pass


class CpoMismatchError(BaseSimulationError):
    """Exception that corresponds to a mismatch between the
    `task`-defined `bodies` list and the cpos to be recorded during
    the simulation."""
    pass


class Simulation(Process):
    """Simulation job."""

    def __init__(self, task, params, info_lock, save=False):
        self.task = task
        self.params = params
        self.info_lock = info_lock

        self.proclabel = None
        self.scene = None
        self.bbase = None
        self.debug_np = None
        self.cache = None
        self.floor = None
        self.cpo = None
        self.posquat_sz = 7
        self.save = save

        self.start_time = None
        self.end_time = None
        self.sim_time = 0

        super(Simulation, self).__init__()

    @contextmanager
    def _sim_context(self, pcpos):
        """Sets up the cpo."""
        tags = ("shape",)

        # Disconnect pcpos from tree.
        parents = []
        for pcpo in pcpos:
            parents.append(pcpo.getParent())
        NodePathCollection(pcpos).wrtReparentTo(self.scene)
        self.scene.init_tree(tags=tags)

        # Add the pcpos to the Bullet world.
        self.bbase.attach(pcpos)
        yield

        # Remove the pcpos from the Bullet world.
        self.bbase.remove(pcpos)
        self.scene.destroy_tree(tags=tags)

        # Reassemble tree.
        for pcpo, parent in zip(pcpos, parents):
            pcpo.wrtReparentTo(parent)

    def _add_noise(self, cpos, pcpos, noises):
        """Adds geometry noise."""
        if (noises == 0).all():
            return

        for cpo, noise in zip(cpos, noises):
            pos = cpo.getPos(self.scene)
            pos += Point3(*noise)
            cpo.setPos(self.scene, pos)

        # Repel.
        with self._sim_context(pcpos):
            self.bbase.repel(50)

    def _prepare_resources(self):
        """Set up all of the nodes and physics resources."""
        # Set up scene.
        self.scene = SSO("scene")
        # Physics.
        self.bbase = BulletBase()
        self.bbase.init()
        self.bbase.gravity = self.params["physics"]["gravity"]
        self.bbase.sim_par = {
            "size": self.params['simulation']["step_size"],
            "n_subs": self.params['simulation']['n_substeps'],
            "size_sub": self.params['simulation']["substep_size"],
        }

    def _prepare_scene(self, cpo_path, floor_path, rec_names):
        """Sets up the current scene."""
        def get_name(cpo):
            return cpo.getName()

        floor = load_cpo(floor_path)
        PSOStyler().apply(floor, "floor")
        floor.reparentTo(self.scene)

        cpo = load_cpo(cpo_path)
        cpo.reparentTo(self.scene)

        self.cache = self.scene.store_tree()
        self.floor = floor
        self.cpo = cpo

        self.scene.init_tree(tags=())
        pcpos = self.scene.descendants(type_=PSO)
        for pcpo in pcpos:
            pcpo.setCollideMask(BitMask32.allOn())
            pcpo.node().setDeactivationEnabled(False)

        cpos_rec = self._order_cpos(
            cpo.descendants(type_=PSO, names=rec_names))

        return pcpos, cpos_rec

    def _clean_scene(self):
        """Clean up the current scene."""
        try:
            self.scene.destroy_tree()
        except AttributeError:
            pass
        try:
            self.cpo.destroy_tree()
        except AttributeError:
            pass
        self.cache = None
        self.scene = None
        self.cpo = None

    def _clean_resources(self):
        """Clean up all resources."""
        self._clean_scene()
        try:
            self.scene.destroy_tree()
        except AttributeError:
            pass
        self.bbase.destroy()

    def _order_cpos(self, cpos_rec0):
        names0 = list(self.task["bodies"])
        names1 = [cpo.getName() for cpo in cpos_rec0]
        if names0 != names1:
            # Names don't match.
            if sorted(names0) == sorted(names1):
                # If they're just out of order, fix them.
                cpos_rec = [cpos_rec0[names1.index(n)] for n in names0]
            else:
                # Otherwise.
                raise CpoMismatchError(
                    "Task bodies and cpos_rec mismatch:\nT: %s\nC: %s" %
                    (", ".join(names0), ", ".join(names1)))
        else:
            cpos_rec = cpos_rec0
        return cpos_rec

    def _set_masses(self, kappa, cpos):
        blocktypes = get_blocktypes(self.cpo)
        style = get_style(path(self.task["cpo_path"]).dirname().name)
        styler = PSOStyler()
        for cpo, blocktype in zip(cpos, blocktypes):
            styler.apply(cpo, style, blocktype=blocktype, kappa=kappa)

    def _simulate(self, data, noise, force, kappa, pcpos, rec_cpos, rec_ints):
        # Get simulation parameters
        step_size = self.bbase.sim_par["size"]
        n_substeps = self.bbase.sim_par["n_subs"]

        # Store pre-noise states
        read(data[0], rec_cpos)

        # Add position noise
        self._add_noise(rec_cpos, pcpos, noise)

        # Store post-noise states
        read(data[1], rec_cpos)

        # Update masses
        self._set_masses(kappa, rec_cpos)

        # Set up force function
        force_dur = self.params['physics']['force_duration']
        force_pcpos = [x.node() for x in rec_cpos]
        force_vecpos = get_force(force['dir'], force['mag'])

        condition_time = 0.
        with self._sim_context(pcpos):
            # Iterative over record intervals.
            for i, interval in enumerate(rec_ints, start=2):

                # Step size and number of substeps for this
                # interval
                size = interval * step_size
                n_subs = interval * n_substeps

                # Set force with appropriate duration.
                dur = min(size, force_dur - condition_time)
                if dur > 0.:
                    tforce = (force_pcpos, force_vecpos, dur)
                else:
                    tforce = None

                # Simulate physics for this recording interval.
                self.bbase.step(size, n_subs, force=tforce)
                condition_time += size

                # Store the cpos' states.
                read(data[i], rec_cpos)

                # Sanity check, to make sure the blocks are all above
                # the floor
                if (data[i][..., 2] < 0).any():
                    mp.util.info("Object z-positions are negative!")
                    print data[i][..., :3]
                    raise Warning("invalid cpo positions")

            self.cache.restore()

        return condition_time

    def simulate_all(self):

        ## Assorted parameters.
        icpo = self.task["icpo"]
        conditions = self.task['conditions']

        ## Set up the cpo.
        pcpos, record_cpos = self._prepare_scene(
            path(self.task["cpo_path"]),
            path(self.task["floor_path"]),
            self.task['bodies'])

        # Determine recording intervals
        record_intervals = self.task['record_intervals']
        # Allocate data storage.
        alldata = np.zeros(self.task['shape'])

        # enumerate over the parameters of each condition
        for icond, cond in enumerate(conditions):
            (iS, S), (iP, P), (iK, K), (isamp, samp) = cond

            data = alldata[icond]
            # shape of noises is (n_sigmas, n_samples, n_objs)
            noise = self.params['noises'][iS, icpo, isamp]
            # shape of forces is (n_forces, n_samples, n_objs)
            force = self.params['forces'][iP, icpo, isamp]

            self.sim_time += self._simulate(
                data, noise, force, float(K), pcpos,
                record_cpos, record_intervals)

        if self.save:
            # Write data to file.
            data_path = path(self.task["data_path"])
            if not data_path.dirname().exists():
                data_path.dirname().makedirs()
            np.save(data_path, alldata)

            # Mark simulation as complete.
            self.task["complete"] = True

        self._clean_scene()

    def print_info(self):
        self.info_lock.acquire()
        n_conditions = len(self.task['conditions'])
        dt = self.end_time - self.start_time
        avg = timedelta(seconds=(dt.total_seconds() / float(n_conditions)))
        speedup = 100 * self.sim_time / dt.total_seconds()

        mp.util.info("-" * 60)
        mp.util.info("Total sim. time   : %s" % str(
            timedelta(seconds=self.sim_time)))
        mp.util.info("Total real time   : %s" % str(dt))
        mp.util.info("Avg. per condition: %s" % str(avg))
        mp.util.info("Num conditions    : %d" % n_conditions)
        mp.util.info("Speedup is %.1f%%" % speedup)
        mp.util.info("-" * 60)

        sys.stdout.flush()
        self.info_lock.release()

    def run(self):
        """Run one simulation."""
        self._prepare_resources()

        try:
            self.start_time = datetime.now()
            self.simulate_all()

        except KeyboardInterrupt:
            mp.util.debug("Keyboard interrupt!")
            sys.exit(100)

        except Exception as err:
            mp.util.debug("Error: %s" % err)
            raise

        else:
            self.end_time = datetime.now()
            # print out information about the task in a thread-safe
            # manner (so information isn't interleaved across tasks)
            self.print_info()

        finally:
            # Clean simulation resources.
            self._clean_resources()
