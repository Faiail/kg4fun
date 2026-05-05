# Wikidata Thematic Subgraph Selection

These datasets have been designed to train and evaluate algorithms to select thematic subgraphs of interest in a large knowledge graph from seed entities of interest.
Specifically, we consider Wikidata. 
Given a set of seed QIDs of interest, a graph expansion is performed following P31, P279, and (-)P279 edges.
Traversed classes that thematically deviates from seed QIDs of interest should be pruned.
Datasets thus consist of classes reached from seed QIDs that are labeled as "to prune" or "to keep".

## Available datasets

| Dataset                   | # Seed QIDs | # Labeled decisions | # Prune decisions | Min prune depth | Max prune depth | # Keep decisions | Min keep depth | Max keep depth | # Reached nodes up | # Reached nodes down |
|---------------------------|-------------|---------------------|-------------------|-----------------|-----------------|------------------|----------------|----------------|--------------------|----------------------|
| [dataset1](data/dataset1) | 455         | 5233                | 3464              | 1               | 4               | 1769             | 1              | 4              | 1507               | 2593609              |
| [dataset2](data/dataset2) | 105         | 982                 | 388               | 1               | 2               | 594              | 1              | 3              | 1159               | 1247385              |


Each dataset folder contains
* ``datasetX.csv``: a CSV file containing one seed QID per line (not the complete URL, just the QID). This CSV file has no header. 
* ``datasetX_labels.csv``: a CSV file containing one seed QID per line and its label (not the complete URL, just the QID)
* ``datasetX_gold_decisions.csv``: a CSV file with seed QIDs, reached QIDs, and the labeled decision (1: keep, 0: prune)
* ``datasetX_Y_folds.pkl``: folds to train and test models based on the labeled decisions

``dataset1-2`` consists of using ``dataset1`` for training and ``dataset2`` for testing.

## License 

Datasets are available under the [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license.



Merge edge types using subproperty of. https://www.wikidata.org/wiki/Property:P1647


Property graph: relations:
    1. Suproperry of
    2. Related (?)
    3. 