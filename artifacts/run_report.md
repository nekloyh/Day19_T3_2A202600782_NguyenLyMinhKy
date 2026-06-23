# GraphRAG run report

- Extractor: `openai`
- Model: `gpt-4o-mini`
- Documents: 70; triples: 331; nodes: 597; edges: 993
- Artifact build time for this run: 0.479s
- API calls: 70; input tokens: 61252; output tokens: 36853; estimated cost: $0.03129960

| System | Top-1 | Hit@3 | MRR | Answer-term coverage |
| --- | ---: | ---: | ---: | ---: |
| Flat RAG (TF-IDF) | 0.30 | 0.45 | 0.457 | 0.40 |
| GraphRAG (entity + 2-hop) | 0.85 | 0.95 | 0.910 | 0.70 |
