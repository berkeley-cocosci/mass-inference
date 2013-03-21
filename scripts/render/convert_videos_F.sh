#!/bin/bash

cmd="convert_videos.py --mp4 --ogg --flv --wmv --dry-run F"

python $cmd --rename=unstable-example stability-example-unstable-F
python $cmd --rename=stable-example stability-example-stable-F
python $cmd --rename=mass-example mass-oneshot-example-F~kappa-1.0
python $cmd mass-oneshot-training-F mass-oneshot-F~kappa-1.0

