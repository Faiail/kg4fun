# !/bin/bash

wget -O ./data/raw/pruning-datasets.zip "https://zenodo.org/records/8091584/files/pruning-datasets.zip?download=1"
unzip ./data/raw/pruning-datasets.zip -d ./data/raw/
mv ./data/raw/data ./data/raw/pruning-datasets