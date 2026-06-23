# Rubric đối chiếu yêu cầu bài lab

Không có thang điểm chính thức trong đề. Bảng này dùng để kiểm tra tất cả deliverable trước khi nộp.

| Hạng mục | Tiêu chí chấm | Bằng chứng trong repository | Trạng thái |
| --- | --- | --- | --- |
| Indexing | Trích xuất thực thể/quan hệ thành triples | `triples.csv`; 70 checkpoint `gpt-4o-mini` có JSON schema/evidence verification | Hoàn thành |
| Provenance | Mỗi triple truy ngược được nguồn | `doc_id`, `sentence/evidence` trong `triples.csv`; unit test provenance | Hoàn thành |
| Construction | Dựng graph và trực quan hóa | `knowledge_graph.graphml`, `knowledge_graph.png`, NetworkX `MultiDiGraph` | Hoàn thành |
| Querying | Entity match, tối đa 2-hop, textualization, citation | `graph_rank`, `answer_with_citation`; output query có triples và `[Nguồn: doc_n]` | Hoàn thành |
| Flat RAG | Baseline độc lập | TF-IDF implementation `flat_rank` | Hoàn thành |
| Evaluation | 20 câu hỏi quan hệ/đa-hop độc lập, gold source/answer, metric | `BENCHMARKS`, `benchmark_20.csv`, `benchmark_summary.csv`, `evaluation_analysis.md` | Hoàn thành |
| So sánh | Báo cáo cả hai hệ thống thắng/thua; không claim quá mức (Graph thắng 13, hòa 7, thua 0) | `evaluation_analysis.md` | Hoàn thành |
| Cost/time | Ghi indexing time, API calls, token và chi phí | `metrics.json`, `token_usage.json`, `cost_analysis.md` | Hoàn thành: 70 API calls, token và cost thực tế |
| Reproducibility | Cài/chạy/test bằng uv | `pyproject.toml`, `uv.lock`, `Makefile`, `README.md`, `test_lab.py` | Hoàn thành |

## Điều kiện để tự nhận “100%” với rubric nghiêm ngặt

Hoàn thành: tất cả 70 checkpoint dùng `gpt-4o-mini`. Có thể chạy lại lệnh `OPENAI_API_KEY=... uv run python graphrag_lab.py --extractor openai --model <model>` để tái tạo artifact từ cache mà không gọi lại API.
