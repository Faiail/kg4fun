# !/bin/bash

echo "Node Types"
uv run main.py summarize --parameters configs/summarization/dataset1/qwen3-8b/node_types.yaml --cls SummarizationRun
echo "Edge Types"
uv run main.py summarize --parameters configs/summarization/dataset1/qwen3-8b/edge_types.yaml --cls SummarizationRun
