# Mass Inference Project

A project exploring how people make inferences about the masses of
objects.

## Repository layout

### Submodules

* [https://github.com/jhamrick/mass-inference-analysis](`analysis`
  submodule): Code that involves analysis of human and/or model data,
  such as using model predictions to select stimuli, or evaluating
  experiment data under the model.

* `data` submodule: Storage of human and model data. Not publicly
  available for privacy reasons.

* [https://github.com/jhamrick/mass-inference-experiment](`experiment`
  submodule): Code to run the experiments.

* [https://github.com/jhamrick/mass-inference-stimuli](`stimuli`
  submodule): Storage of stimuli in various forms, including code
  objects and rendered video.

### Directories

* `scripts`: Various utility scripts to help generate and manage
  stimuli and data.

* `scripts/render`: Scripts to view stimuli and render videos of them.

* `scripts/simulate`: Scripts to generate raw IPE ("Intuitive Physics
  Engine") simulations.

* `scripts/util`: Module containing helper functions for programs in
  `scripts`.

