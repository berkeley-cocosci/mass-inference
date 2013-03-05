# Mass Learning: Experiment

## Structure

* `config` -- trial metadata and order for each condition
* `config/hits` -- Mechanical Turk HITs
* `data` -- auto-generated folder with participant data
* `db_tools.py` -- helper functions for interacting with the databases
* `gen_conditions.py` -- script for generating experimental conditions
* `index.py` -- server-side experiment code
* `logs` -- auto-generated folder with experiment logs
* `resources` -- static or client-side resources
* `resources/css/experiment.css` -- all CSS formatting
* `resources/flowplayer` -- video player, see [http://flowplayer.org/](http://flowplayer.org/)
* `resources/html/experiment.html` -- HTML template code
* `resources/images` -- non-stimuli images
* `resources/js/experiment.js` -- experiment JavaScript

## Setup

To run the experiment, you'll need to do two things in particular.

1. **Create a symlink to *stimuli*.** This requires access to the mass
   learning stimuli repository. If you have this repository checked
   out in a folder called `stimuli` at the same level as this
   repository (i.e., `../stimuli`), then the following command should
   work:

	`ln -s ../stimuli/www stimuli`

2. **Initialize the database.** The database stores IP addresses,
   validation codes, completion codes, conditions, and participant
   ids. It is not included in the repository for privacy reasons and
   should never be committed; you will need to manually initialize
   it. To do this:

	`$ python -i db_tools.py  
	create()  
	Backing up old database to 'data/data.db.bak'...  
	Creating new database 'data/data.db'...  
	Created 'Participants' table`
