#!/bin/bash

mkdir _build
cp -r tct ./_build/tct
rm -r ./_build/tct/__pycache__
python -m zipapp --main tct.tou_batch:main _build
mv _build.pyz tou_batch.pyz
rm -r ./_build