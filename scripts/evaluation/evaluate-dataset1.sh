# !/bin/bash

echo "Node Types"
uv run main.py evaluate --parameters configs/evaluation/dataset1/node_types/labels.yaml --cls EvaluationRun
uv run main.py evaluate --parameters configs/evaluation/dataset1/node_types/descriptions.yaml --cls EvaluationRun

echo "Edge Types"
uv run main.py evaluate --parameters configs/evaluation/dataset1/edge_types/labels.yaml --cls EvaluationRun
uv run main.py evaluate --parameters configs/evaluation/dataset1/edge_types/descriptions.yaml --cls EvaluationRun