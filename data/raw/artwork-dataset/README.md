# Wikidata Thematic Subgraph Selection - The case of Artworks


## Dataset Description
This dataset complements the two datasets presented in [Wikidata Thematic Subgraph Selection](/data/raw/pruning-datasets/README.md).

Specifically, this dataset was built starting from a set of seed QIDs of interest from the domain of *Art and Works od Art*. A graph expansion is performed following edges in Wikidata. Trasversed entitities that deviates from the thematic subgraph should be pruned. The dataset thus consists of annotated pairs of entities `(seed, neighbor)`, identified by their Wikidata QIDs, along with a binary decision: *keep* or *prune*.


## Dataset Metadata
The structure of the dataset is the same introduced in [Wikidata Thematic Subgraph Selection](/data/raw/pruning-datasets/README.md).