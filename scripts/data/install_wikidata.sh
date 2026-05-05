# !/bin/bash

source .venv/bin/activate
mkdir -p ./data/external/wikidata && cd ./data/external/wikidata
qlever setup-config wikidata
qlever get-data
qlever index
qlever start
qlever ui
qlever autocompletion-warmup
cd ../../../