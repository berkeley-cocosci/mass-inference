#!/bin/sh -e

DATA_PATH=data
git annex add $DATA_PATH/human-raw/mass_inference-I/
git annex add $DATA_PATH/human/mass_inference-I.dpkg/*.csv
git add $DATA_PATH/human/mass_inference-I.dpkg/datapackage.json
git annex add $DATA_PATH/human/mass_inference-merged.dpkg/*.csv
git add $DATA_PATH/human/mass_inference-merged.dpkg/datapackage.json
git annex add $DATA_PATH/human/workers.db

