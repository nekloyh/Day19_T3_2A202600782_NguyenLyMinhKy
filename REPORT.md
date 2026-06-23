# Báo cáo lab GraphRAG

## Phạm vi đã triển khai

1. Đọc 70 tài liệu trong `dataset/`, trích xuất triples có predicate và provenance ở cấp câu.
2. Dựng `NetworkX.MultiDiGraph`; xuất GraphML và một đồ thị con có relation labels.
3. Query theo entity → fact/source tối đa hai hop; textualize bằng câu nguồn và luôn trả `[Nguồn: doc_n]`.
4. So sánh GraphRAG hybrid với Flat RAG TF-IDF qua 20 câu hỏi độc lập có gold source/answer.
5. Xuất metrics, token usage, cost analysis và phân tích các ca hệ thống thắng/thua.

## Kết quả lần chạy OpenAI đã kiểm chứng

| Hệ thống | Top-1 | Hit@3 | MRR | Answer-term coverage |
| --- | ---: | ---: | ---: | ---: |
| Flat RAG (TF-IDF) | 0.35 | 0.55 | 0.467 | 0.20 |
| GraphRAG (entity + 2-hop) | 0.40 | 0.55 | 0.520 | 0.20 |

GraphRAG tăng Top-1 và MRR trong lần chạy này, nhưng Flat RAG cao hơn ở Hit@3. Vì vậy kết luận đúng là: **GraphRAG cải thiện thứ hạng đầu trong benchmark này, không thắng ở mọi metric**. Các ca khác biệt được ghi trong `artifacts/evaluation_analysis.md`.

## Chi phí và thời gian

- 70 documents, 597 nodes, 993 edges, 331 semantic triples.
- Full OpenAI extraction với `gpt-4o-mini`: 70 API calls, 61,252 input tokens, 36,853 output tokens, chi phí $0.03129960.
- Lượt index LLM hoàn chỉnh mất 800.173 giây. Các lượt chạy lại từ checkpoint chỉ rebuild artifact và không tạo thêm extraction call.

## Bảo đảm chất lượng

- `test_lab.py` xác thực số lượng corpus/benchmark, provenance của triple và query graph trả evidence.
- Extractor LLM loại mọi triple không có `evidence` nguyên văn trong source document.
- Benchmark không dùng 7 trường `Query` bị lặp của dataset; mỗi case có target source, gold answer và required terms độc lập với title.

## Chạy nộp bài

```bash
make install
make test
make run
```

Để dùng LLM thật, cần cấu hình API key và mức giá model hiện hành như trong README, rồi chạy `uv run python graphrag_lab.py --extractor openai --model <model>`.
