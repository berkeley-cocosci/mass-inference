from itertools import product as iproduct
from path import path
import json
import numpy as np
from mass import CPO_PATH


class Tasks(dict):

    def save(self, filename):
        with open(path(filename), "w") as fh:
            json.dump(self, fh)

    @classmethod
    def load(cls, filename):
        tasks = cls()
        with open(path(filename), "r") as fh:
            tasks.update(json.load(fh))
        return tasks

    @classmethod
    def create(cls, params):
        """Create the tasks dictionary from the parameters."""

        sim_root = path(params["sim_root"])
        if not sim_root.exists():
            sim_root.makedirs_p()

        floor_path = CPO_PATH.joinpath(params['floor_path'])
        cpo_paths = [CPO_PATH.joinpath(x) for x in params['cpo_paths']]

        index_names = params['index_names']
        index_levels = params['index_levels']
        cpos_rec_names = index_levels['object']
        record_intervals = list(np.diff(index_levels['timestep'][1:]))

        cond_names = [
            'sigma',
            'phi',
            'kappa',
            'stimulus',
            'sample',
        ]

        conditions = list(iproduct(*[
            enumerate(index_levels[x]) for x in cond_names
            if x != 'stimulus']))
        n_conditions = len(conditions)
        n_chunks = int(np.ceil(n_conditions / float(params['max_chunk_size'])))
        chunks = np.array_split(np.arange(n_conditions), n_chunks, axis=0)
        base_shape = [
            len(index_levels[x]) for x in index_names
            if x not in cond_names]

        tasks = cls()
        completed = cls()
        for icpo, cp in enumerate(cpo_paths):
            for ichunk, chunk_idx in enumerate(chunks):
                sim_name = "%s_%s_%02d" % (cp.namebase, params["tag"], ichunk)
                data_path = sim_root.joinpath("%s.npy" % sim_name)
                chunk = [conditions[i] for i in chunk_idx]
                shape = [len(chunk)] + base_shape

                # Make the task dicts for this sample.
                tasks[sim_name] = {
                    "icpo": icpo,
                    "floor_path": str(floor_path),
                    "cpo_path": str(cp),
                    "data_path": str(data_path),
                    "script_root": params["script_root"],
                    "task_name": sim_name,
                    "bodies": cpos_rec_names,
                    "seed": abs(hash(sim_name)),
                    "conditions": chunk,
                    "record_intervals": record_intervals,
                    "shape": shape,
                    "num_tries": 0,
                }

                completed[sim_name] = False

        return tasks, completed
