# Experiment

Scripts to manage experiment stimuli files and conditions.

* `convert_videos.py`: convert stimuli videos into various web formats
  and copy image files into the appropriate location. Takes a target
  condition `TARGET`, and copies/converts from
  `experiment/render/TARGET` to `stimuli/www/TARGET/images` and
  `stimuli/www/TARGET/video`.

  * `convert_videos_F.sh`: copy/convert stimuli for experiment version F.

* `gen_conditions.py`: generate the trial list/config for an
  experimental condition. The condition name should have the syntax
  `EXP_VER-FBTYPE-RATIO-CB`, where `EXP_VER` is the experiment version
  (e.g. 'F'), `FBTYPE` is the feedback type (e.g. 'fb', 'vfb', 'nfb'),
  `RATIO` is the mass ratio (e.g. '10'), and `CB` is whether the
  condition is a counterbalanced condition (valid values are 'cb0' or
  'cb1').

  * `gen_conditions_F.sh`: generate conditions for experiment version F.

* `deploy`: a script to deploy the experiment to the webserver. Takes
  a single argument, `EXP_VER`, which specifies the experiment
  version, and then uses rsync to copy over the experiment to
  `PhysicsExperiment-EXP_VER`.
