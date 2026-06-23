# GraphRAG Knowledge Graph Lab

Lab hoàn chỉnh theo 4 bước: trích xuất triples, dựng knowledge graph NetworkX, truy vấn entity + 2-hop, và benchmark Flat RAG với GraphRAG.

## Chạy

```bash
uv sync
uv run python graphrag_lab.py
uv run python graphrag_lab.py --query "What policies support electric vehicle market growth in US cities?"
uv run --env-file .env python graphrag_lab.py --extractor openai --model gpt-4o-mini --query "How many U.S. buyers chose EVs in 2023?" --answer-with-llm
uv run python -m unittest -v test_lab.py
```

Kết quả được tạo trong `artifacts/`:

- `triples.csv`: các triples kèm document và câu bằng chứng.
- `knowledge_graph.graphml`: graph mở được bằng Gephi hoặc Neo4j import tooling.
- `knowledge_graph.png`: ảnh trực quan hóa graph.
- `benchmark_20.csv` và `benchmark_summary.csv`: so sánh 20 câu hỏi Flat RAG/GraphRAG.
- `evaluation_analysis.md`: các ca GraphRAG/Flat RAG thắng-thua và giới hạn diễn giải.
- `run_report.md`, `cost_analysis.md`, `metrics.json`, `token_usage.json`: thời gian, số node/edge/triple, token và chi phí của lần chạy.

## Thiết kế

Chế độ mặc định `heuristic` không gọi API. Nó tạo các predicate bảo thủ như `REPORTS_FINANCIAL_RESULT`, `REPORTS_SALES`, `REPORTS_POLICY`, `REPORTS_FORECAST` khi câu nguồn chứa tín hiệu tương ứng; các câu khác chỉ tạo cạnh `MENTIONS`/`RELATED_TO`. Mọi cạnh đều mang `doc_id` và câu `evidence` nguyên văn. Truy vấn xác định entity, duyệt graph tối đa hai hop có provenance và dùng lexical retrieval làm fallback khi không có entity đủ đặc hiệu. Khi chạy `--answer-with-llm`, generator chỉ nhận context từ ba source GraphRAG đầu và phải trích dẫn `[doc_n]` cho từng fact.

Để trích xuất quan hệ ngữ nghĩa bằng LLM:

```bash
export OPENAI_API_KEY='...'
export OPENAI_INPUT_COST_PER_1M='...'   # giá hiện hành của model, USD/1M input tokens
export OPENAI_OUTPUT_COST_PER_1M='...'  # giá hiện hành của model, USD/1M output tokens
uv run python graphrag_lab.py --extractor openai --model gpt-4o-mini
```

Không commit API key. Chế độ này thực hiện một lời gọi trích xuất cho mỗi tài liệu, từ chối mọi triple không kèm evidence nguyên văn có trong tài liệu, và lưu token usage/cost thực tế của API response. Mặc định chạy một worker và chờ 15 giây giữa các request để phù hợp quota thấp. Các kết quả thành công được checkpoint trong `artifacts/llm_cache/`; chạy lại sẽ chỉ gọi các tài liệu chưa hoàn thành. Tài khoản quota cao có thể chỉnh `GRAPHRAG_LLM_WORKERS` và `GRAPHRAG_LLM_REQUEST_DELAY`.

Nếu API quota bị giới hạn giữa chừng, có thể vẫn dựng artifact hoàn chỉnh từ các checkpoint LLM đã có và fallback heuristic cho phần còn lại:

```bash
uv run python graphrag_lab.py --extractor hybrid --model gpt-4o-mini
```

## Giới hạn đánh giá

Nguồn dữ liệu chỉ có 7 trường `Query` lặp lại trên 70 tài liệu, nên chúng không thể xác định duy nhất một `expected_doc`. Lab dùng 20 câu hỏi độc lập với tiêu đề; mỗi case có gold source, gold answer và required answer terms trong `benchmark_20.csv`. Báo cáo đo retrieval (`Top-1`, `Hit@3`, `MRR`) và term coverage của câu trả lời trích dẫn. Đây không phải phép đo hallucination của mô hình sinh tự do; muốn đánh giá hallucination cần chạy generator LLM và dùng chấm điểm độc lập.
