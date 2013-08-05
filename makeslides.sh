#!/bin/sh -e

cd man/$1
ipython nbconvert --to slides $1.ipynb
