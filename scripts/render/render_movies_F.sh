#!/bin/bash -e

python render_movies.py --stype=stability-example-unstable-F --original F
python render_movies.py --stype=stability-example-stable-F --original F
python render_movies.py --stype=mass-oneshot-training-F --original --feedback F
python render_movies.py --stype=mass-oneshot-example-F --inference --counterbalance F mass-oneshot-example-F~kappa-1.0
python render_movies.py --stype=mass-oneshot-F --inference --counterbalance --feedback F mass-oneshot-F~kappa-1.0
python render_movies.py --stype=mass-oneshot-F --inference F-demo mass-oneshot-F~kappa-1.0

