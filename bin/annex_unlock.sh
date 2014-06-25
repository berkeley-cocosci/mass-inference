#!/bin/sh -e 

DATA_PATH=data
git annex unlock $DATA_PATH/human-raw/mass_inference-I/
git annex unlock $DATA_PATH/human/mass_inference-I.dpkg
git annex unlock $DATA_PATH/human/mass_inference-merged.dpkg
git annex unlock $DATA_PATH/human/workers.db
