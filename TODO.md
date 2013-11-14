# To-do lists

## Before running the pilot

* TODO: figure out why it's trying to load an unsafe script
* TODO: test all conditions of experiment
* TODO: make sure IE users are locked out
* TODO: drop `mass-inference-G-test` table from database

## Running pilot

* TODO: set up experiment on the sandbox
* TODO: collect some pilot data in sandbox from labmates

## After running the pilot

* TODO: write tentative analysis of experiment data
* TODO: verify that experiment actually takes about 15 minutes

## Before running the actual experiment

* TODO: set database table to `mass-inference-G`
* TODO: set `$c.debug` in `config.js` to false
* TODO: enable `psiTurk.finishInstructions` in `task.js`

## Other stuff

* TODO: comment experiment javascript code
* TODO: write some unit tests
* TODO: add a way to notify server if client errors during simulations

## Simulations

[x] stability_original
	[x] ipe
	[x] truth
[x] stability_sameheight
	[x] ipe
	[x] truth
[ ] mass_all
	[ ] ipe
	[x] truth
[x] mass_learning
	[x] ipe
	[x] truth
[ ] mass_prediction_stability
	[ ] ipe
	[x] truth
[ ] mass_prediction_direction
	[ ] ipe
	[ ] truth
[x] mass_inference-G-a
	[x] ipe
	[x] truth
[x] mass_inference-G-b
	[x] ipe
	[x] truth
