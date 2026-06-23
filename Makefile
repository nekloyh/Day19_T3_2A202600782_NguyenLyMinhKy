.PHONY: install test run query

install:
	uv sync

test:
	uv run python -m unittest -v test_lab.py

run:
	uv run python graphrag_lab.py

query:
	uv run python graphrag_lab.py --query "What policies support electric vehicle market growth in US cities?"
