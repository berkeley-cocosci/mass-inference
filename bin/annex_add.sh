#!/bin/sh -e

DATA_PATH=data
git annex add resources/sso/metadata.db
git annex add resources/sim-scripts/mass_inference-I*
git annex add --force $DATA_PATH/sim-raw/mass_inference-I*/*.tar.gz
git annex add $DATA_PATH/model-raw/mass_inference-I*/*.npy
git add $DATA_PATH/model-raw/mass_inference-I*/datapackage.json
git annex add $DATA_PATH/model/mass_inference-I*/*.csv
git add $DATA_PATH/model/mass_inference-I*/datapackage.json
