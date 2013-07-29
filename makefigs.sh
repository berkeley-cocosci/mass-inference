#!/bin/sh -e

# generate OmniGraffle figures
for fig in figures/*.graffle; do
    echo "Exporting '$fig'..."
    omnigraffle-export -f png "$fig" figures/
done
/usr/bin/osascript -e 'tell application "OmniGraffle Professional 5" to quit'

# now run analyses and generate figures from them
cd "analysis"
python ../run_nb.py analyze-stability-learning.ipynb
python ../run_nb.py cogsci2013-poster-figures.ipynb
