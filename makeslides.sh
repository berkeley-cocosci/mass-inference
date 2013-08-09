#!/bin/sh -e

cd man/$1
ipython nbconvert --RevealHelpTransformer.url_prefix=reveal.js --to slides $1.ipynb
