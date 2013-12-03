# Mass Inference Project

A project exploring how people make inferences about the masses of
objects.

## Choosing stimuli

First, pick stimuli by executing code in the notebook
`lib/mass/analysis/choose-stimuli.ipynb`. This will create new
directories in `resources/sso` for the desired stimuli.

## Running simulations

You can run most everything with the script `bin/simulate.py`, though
if you need finer-grained control over individual steps of the
simulations, you can use the scripts in `bin/simulate`. For each step,
both options are listed, or you can run all steps at once using
`bin/simulate.py --all`.

Note, however, that the client will still need to be run separately:
if you run `bin/simulate.py --all`, when it gets to the server, it
will run the server and wait for a client to connect. Once all
simulations are run and the server exits, it will move on to
processing the simulations.

1. First you need to generate sim scripts:

    `bin/simulate.py -e mass_inference -t G-b-truth --generate`
    `bin/simulate/generate_script.py -e mass_inference -t G-b-truth`.

2. Then, launch the server with the appropriate parameters for the
   simulation, e.g.:

    `bin/simulate.py -e mass_inference -t G-b-truth --run-server`
	`bin/simulate/run_simulations.py server -e mass_inference -t G-b-truth -k hello -f`

3. Then run the client, e.g.:

    `bin/simulate.py -e mass_inference -t G-b-truth --run-client`
	`bin/simulate/run_sims.py client -k hello -s -n 2`

4. Finally, process the simulations and save them as datapackages:

    `bin/simulate.py -e mass_inference -t G-b-truth --process`
    `bin/simulate/process_simulations.py -e mass_inference -t G-b-truth`

## Computing model queries

TODO: more details on computing model queries

1. Run `bin/process_model_fall.py`
2. Run `bin/save_stability.py`

## Rendering stimuli

You can run most everything with the script `bin/render.py`, though if
you need finer-grained control over individual steps of the render,
you can use the scripts in `bin/render`. For each step, both options
are listed, or you can run all steps at once using `bin/render.py
--all`.

1. First create the rendering scripts:

	`bin/render.py -e mass_inference-G --generate`
	`bin/render/generate_script.py -e mass_inference-G`

2. Then run the renderer with the appropriate parameters, e.g.:

	`bin/render.py -e mass_inference-G --render`
	`bin/render/render_stimuli.py -e mass_inference-G`

3. Now convert the videos that were rendered to various webformats:

	`bin/render.py -e mass_inference-G --convert`
	`bin/render/convert_videos.py -e mass_inference-G`

## Deploying the experiment

TODO: more details on deploying the experiment

1. Run `bin/experiment/link_stimuli.py`
2. Run `bin/experiment/generate_configs.py`
3. Run `bin/experiment/deploy_experiment.py`
