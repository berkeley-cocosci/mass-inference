# Mass Inference Project: Utility Scripts

## Directories

* `scripts/render`: Scripts to view stimuli and render videos of them.

* `scripts/simulate`: Scripts to generate raw IPE ("Intuitive Physics
  Engine") simulations.

* `scripts/util`: Module containing helper functions for programs in
  `scripts`.

## Scripts

* `process_human_data.sh`: Performs all functions needed to put human
  data in a form for analysis. This includes running
  `find_valid_pids.py`, `parse_human_data.py`, and
  `consolidate_human_data.py`.

    - `find_valid_pids.py`: Finds the pids that correspond to
      participants who have started the experiment for the first time
      and which have also completed the experiment (i.e., excludes
      people who have done any amount of the experiment before, or who
      did not finish the experiment). The array of valid pids is
      stored in `data/human/valid_pids.npy`.

    - `parse_human_data.py`: Converts the raw experiment csv files in
      `data/human/raw_data` into NumPy arrays (one array for each
      participant). Data is saved to `data/human/processed_data`.

    - `consolidate_human_data.py`: Consolidates the individual
      participant NumPy arrays in `data/human/processed_data` into
      arrays for each condition and experiment phase (e.g., one array
      for each phase of each condition, but each array includes all
      participants in that condition). Data is saved to
      `data/human/consolidated_data`.

* `process_model_data.py`: Compresses model data in
  `data/sims/compressed` further by computing specified features of
  the data (e.g., number of blocks that fell) and saves the data to
  `data/model`.

* `process_old_data.py`: (requires `cogphysics` library) Compresses
  old model data by computing features of the data (currently just
  number of blocks that fell) and saves model data to
  `data/model`. Loads and resaves old (stability) human data as the
  average data over participants, raw human data, and stimuli; data is
  saved into `data/old-human`.
