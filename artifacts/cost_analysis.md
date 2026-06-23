# Cost analysis

- Extractor: `openai`
- Artifact build time from checkpoints: 0.488s
- Full OpenAI extraction time: 800.173s (see `full_llm_indexing.md`)
- Documents: 70; triples: 331
- LLM calls: 70; input tokens: 61252; output tokens: 36853.
- Estimated cost: $0.03129960.

Cost is calculated from the actual API usage stored in the LLM checkpoint. The standard gpt-4o-mini rates used for this run are $0.15 per 1M input tokens and $0.60 per 1M output tokens, as configured at execution time. The exact provider usage is stored in `token_usage.json`.
