# Evaluation analysis

## Method

Twenty title-independent questions have a manually verified target document, expected answer, required answer terms, and cited source evidence.

## Retrieval comparison

| System | Top-1 | Hit@3 | MRR | Answer-term coverage |
| --- | ---: | ---: | ---: | ---: |
| Flat RAG (TF-IDF) | 0.30 | 0.45 | 0.457 | 0.40 |
| GraphRAG (entity + 2-hop) | 0.85 | 0.95 | 0.910 | 0.70 |

## Cases where GraphRAG recovered a source Flat RAG missed

- B02: Flat rank 6; Graph rank 1. Which Chinese automaker surpassed Tesla as the top-selling electric-car maker in late 2024?
- B03: Flat rank 5; Graph rank 1. Besides Tesla, which two legacy U.S. automakers are listed as members of the S&P 500 index?
- B05: Flat rank 17; Graph rank 1. How much EV-manufacturing investment has Georgia attracted, leading all U.S. states?
- B06: Flat rank 6; Graph rank 2. How much is General Motors investing to launch its EV line-up?
- B09: Flat rank 9; Graph rank 1. Whose EV sales did European EV sales outstrip for the first time in years during 2020?
- B10: Flat rank 4; Graph rank 1. Who became China's minister of science and technology in 2007 and boosted its EV industry?
- B13: Flat rank 5; Graph rank 1. Which company predicted in early 2023 that U.S. EV sales would surpass the one-million mark?
- B17: Flat rank 4; Graph rank 1. Which Department of Energy initiative will provide over $13 billion to improve U.S. grid reliability?
- B19: Flat rank 6; Graph rank 1. Where does the EIA make its open-source code available to the public?
- B20: Flat rank 13; Graph rank 1. Alongside the Inflation Reduction Act's tax credits, which state announced a 2035 ban on new combustion-engine cars?

## Cases where Flat RAG recovered a source GraphRAG missed

No Flat-RAG-only Hit@3 cases in this run.

## Interpretation

The graph was built from LLM-extracted semantic triples with sentence-level provenance. These metrics evaluate retrieval and source-grounded extraction; a separate LLM answer generator is available for manual queries and must be independently judged for hallucination.

## Scope and limitations

- The benchmark is deliberately relation-centric (parent company, index membership, partnership, ranking, attribution), the class of question where entity-anchored traversal is expected to help. On plain one-hop factoid lookups the two systems are typically closer; this run measures the relational regime, not all query types.
- GraphRAG leads with the graph only when a query entity matches a node that has edges (graph 0.6 / lexical 0.4); with no entity match it falls back to lexical TF-IDF. So its advantage depends on entity matching, which in turn depends on extraction quality.
- The tokenizer keeps numeric terms (`$31.2 billion`, `2024`, `$13 billion`) but retrieval has no semantic embedding; a relation phrased with no matchable entity (e.g. asking for a US state without naming it) still falls back to lexical and can rank low.
- The reported answer is a single extracted sentence chosen by question overlap and intent, so answer-term coverage is a strict lower bound on extraction quality, not a free-form generation score.
