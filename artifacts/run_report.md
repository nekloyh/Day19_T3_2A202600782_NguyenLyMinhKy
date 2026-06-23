# GraphRAG run report

- Extractor: `openai`
- Model: `gpt-4o-mini`
- Documents: 70; triples: 331; nodes: 597; edges: 993
- Artifact rebuild time from checkpoints: 0.488s
- Full OpenAI extraction time: 800.173s (see `full_llm_indexing.md`)
- API calls: 70; input tokens: 61252; output tokens: 36853; estimated cost: $0.03129960

| System | Top-1 | Hit@3 | MRR | Answer-term coverage |
| --- | ---: | ---: | ---: | ---: |
| Flat RAG (TF-IDF) | 0.35 | 0.55 | 0.467 | 0.20 |
| GraphRAG (entity + 2-hop) | 0.40 | 0.55 | 0.520 | 0.20 |
