#!/bin/bash -e

python record_playback.py --stype=mass-oneshot-example-F mass-oneshot-example-F~kappa--1.0 mass-oneshot-example-F~kappa-1.0
python record_playback.py --stype=mass-oneshot-F mass-oneshot-F~kappa--1.0 mass-oneshot-F~kappa-1.0
python record_playback.py --stype=mass-oneshot-training-F --original
python record_playback.py --stype=stability-example-stable-F --original
python record_playback.py --stype=stability-example-unstable-F --original


