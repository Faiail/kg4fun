# !/bin/bash

echo "Node Types"
uv run main.py evaluate --parameters configs/evaluation/dataset2/node_types/labels.yaml --cls EvaluationRun
uv run main.py evaluate --parameters configs/evaluation/dataset2/node_types/descriptions.yaml --cls EvaluationRun

echo "Edge Types"
uv run main.py evaluate --parameters configs/evaluation/dataset2/edge_types/labels.yaml --cls EvaluationRun
uv run main.py evaluate --parameters configs/evaluation/dataset2/edge_types/descriptions.yaml --cls EvaluationRun