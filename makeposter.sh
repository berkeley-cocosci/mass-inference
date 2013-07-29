#!/bin/sh -e

poster=$1

# generate OmniGraffle figures
echo "Exporting '$poster'..."
omnigraffle-export -f pdf "$poster" man/
/usr/bin/osascript -e 'tell application "OmniGraffle Professional 5" to quit'
