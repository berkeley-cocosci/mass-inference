#!/bin/sh -e 

DATA_PATH=data
git annex unlock $DATA_PATH/human/mass_inference-G.dpkg
git annex unlock $DATA_PATH/human/mass_inference-H.dpkg
git annex unlock $DATA_PATH/human/mass_inference-I.dpkg
git annex unlock $DATA_PATH/human/mass_inference-merged.dpkg
