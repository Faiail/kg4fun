# Wikidata Thematic Subgraph Selection


## Dataset Description
The datasets have been designed to train and evaluate algorithms to select thematic subgraphs in a large knowledge graph rom seed entities of interest.
Specifically, the authors consider Wikidata as a starting point. Fiven a set of seed QIDs, a graph expansion is performed following P31, P279 and (-)P279 edge types.
The authors aim to prune trasversal classes that deviates from the semantic core expressed by the QIDs.
To this end, the dataset consistas of of classes reached from seed QIDs that are labeled as "*to_prune*" or "*to_keep*"

## Dataset Metadata
Each of **dataset1** or **dataset2** consists of the following data:
 - **`datasetX.csv`**: a CSV file containing one seed QID (no header); (examples are [dataset1.csv](./dataset1/dataset1.csv) or [dataset2.csv](./dataset2/dataset2.csv))
 - **`datasetX_labels.csv`**: a CSV file containing one seed QID per line and its label; (examples are [dataset1_labels.csv](./dataset1/dataset1_labels.csv) or [dataset2_labels.csv](./dataset2/dataset2_labels.csv))
    - **QID**: the seed QID
    - **label**: the label retrieved using the Wikidata property
 - **`datasetX_gold_decisions.csv`**: a CSV file with seed QIDs, reached QIDs, and tghe labeled decision (1: *keep*, 0: *prune*) (exampels are [dataset1_gold_decisions.csv](./dataset1/dataset1_gold_decisions.csv) or [dataset2_gold_decisions.csv](./dataset2/dataset2_gold_decisions.csv))
    - **from**: the head QID - **SEED QID**(?);
    -  **starting_label**: the label of **from** (?);
    - **QID**: the tail qid (?);
    - **label**: the label of **QID** (?);
    - **depth**: the number of nodes (-1) that connect **from** to **QID** (?);
    - **target**: the golden decision that should be made (1: *keep*, 0: *prune*);

**NOTE**: there is the possibility to cross validate the dataset by using the folder [dataset1-2](./dataset2/)
