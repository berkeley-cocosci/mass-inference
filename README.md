# Mass Inference Project

A project exploring how people make inferences about the masses of
objects.

## Repository layout

### Submodules

* [`analysis`](https://github.com/jhamrick/mass-inference-analysis):
  Code that involves analysis of human and/or model data, such as
  using model predictions to select stimuli, or evaluating experiment
  data under the model.

* `data`: Storage of human and model data. Not publicly
  available for privacy reasons.

* [`experiment`](https://github.com/jhamrick/mass-inference-experiment):
  Code to run the experiments.

* [`stimuli`](https://github.com/jhamrick/mass-inference-stimuli):
  Storage of stimuli in various forms, including code objects and
  rendered video.

### Directories

* [`scripts`](https://github.com/jhamrick/mass-inference/tree/master/scripts):
  Various utility scripts to help generate and manage stimuli and
  data.

* [`scripts/render`](https://github.com/jhamrick/mass-inference/tree/master/scripts/render):
  Scripts to view stimuli and render videos of them.

* [`scripts/simulate`](https://github.com/jhamrick/mass-inference/tree/master/scripts/simulate):
  Scripts to generate raw IPE ("Intuitive Physics Engine")
  simulations.

* [`scripts/util`](https://github.com/jhamrick/mass-inference/tree/master/scripts/util):
  Module containing helper functions for programs in `scripts`.

