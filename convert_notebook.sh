#!/usr/bin/env bash


# convert all notebooks to markdown
jupyter nbconvert --to markdown */*/*.ipynb

# convert all notebooks to python scripts
jupyter nbconvert --to script */*/*.ipynb
