#!/bin/sh -e 

DATA_PATH=data
git annex unlock resources/sso/metadata.db
git annex unlock resources/sim-scripts/mass_inference-I*
git annex unlock $DATA_PATH/sim-raw/mass_inference-I*
git annex unlock $DATA_PATH/model-raw/mass_inference-I*
git annex unlock $DATA_PATH/model/mass_inference-I*
