# Evaluation analysis

## Method

Twenty title-independent questions have a manually verified target document, expected answer, required answer terms, and cited source evidence.

## Retrieval comparison

| System | Top-1 | Hit@3 | MRR | Answer-term coverage |
| --- | ---: | ---: | ---: | ---: |
| Flat RAG (TF-IDF) | 0.35 | 0.55 | 0.467 | 0.20 |
| GraphRAG (entity + 2-hop) | 0.40 | 0.55 | 0.520 | 0.20 |

## Cases where GraphRAG recovered a source Flat RAG missed

No GraphRAG-only Hit@3 cases in this run. This is a valid outcome, not a failure to report.

## Cases where Flat RAG recovered a source GraphRAG missed

No Flat-RAG-only Hit@3 cases in this run.

## Interpretation

The graph was built from LLM-extracted semantic triples with sentence-level provenance. These metrics evaluate retrieval and source-grounded extraction; a separate LLM answer generator is available for manual queries and must be independently judged for hallucination.
