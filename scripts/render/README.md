# Viewing and Rendering Stimuli

Scripts to view stimuli and render videos of them.

## Scripts

### Viewing stimuli

All "viewer" scripts have the same argument structure (see
`arghelper.py`). The two most important arguments are `--stype` and
`--playback`; pass the `-h` flag to see more usage information.

The "stype" argument denotes the stimulus type, which is the name of
the directory in `stimuli/obj/old` containing the stimuli you wish to
view. To view specific stimuli and/or an order of the stimuli, you can
pass their names on the command line, or give the name of a list
contained in `stimuli/lists`.

By default, physics will be computed on-the-fly. However, if the
"physics" flag is passed, then the simulations will be loaded from a
*playback* file (contained in `data/playback`). See "Recording
playback" below for information about creating these playback files.

* `view_original_towers.py`: view stimuli as "original" tower scenes
  (randomly colored blocks that all have the same mass)

* `view_prediction_towers.py`: view stimuli as mass "prediction" tower
  scenes (heavy gray blocks and light green blocks, with a 10:1 mass
  ratio by default).

* `view_inference_towers.py`: view stimuli as mass "inference" tower
  scenes (yellow and red blocks, where the mass ratio is yellow:red).

* `view_towers_base.py`: base class which viewers inherit from (not
  actually a script).


### Recording playback

* `record_playback.py`: record "playback" physics simulations. Pass
  `-h` to see usage information.

    * `record_playback_F.sh`: record all playback for stimuli for
      experiment version F.


### Rendering videos

* `render_movies.py`: render movies of stimuli for use in the
  web-based experiment. Variables such as camera angle must be
  specified in a "rendering info" CSV file called
  `stimuli/meta/<target>-rendering-info.csv`, where `<target>` is
  passed as an argument on the command line. Pass `-h` to see other
  usage information.

    * `render_movies_F.sh`: render all stimuli for experiment version
      F.

    * `render_movies_F-demo.sh`: render demo movies for experiment
      version F.


## Other files/modules

* `arghelper.py`: common argument parser for the "viewer" scripts.

* `scenes`: module containing "scene" classes that handle displaying
  the tower scenes in different ways, e.g. "original", "prediction",
  "inference", etc.

