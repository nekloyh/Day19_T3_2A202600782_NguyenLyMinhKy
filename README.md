# Báo cáo Day 19: GraphRAG Knowledge Graph

**Học viên:** Nguyễn Lý Minh Kỳ
**MSSV:** 2A202600782

## 1. Câu hỏi nghiên cứu

**Trích xuất thực thể.** LLM có thể phân biệt thực thể với thuộc tính bằng cách sinh các triple có kiểu: tên nút nằm trong `subject` và `object`, còn thông tin mô tả trở thành nhãn quan hệ, bằng chứng, hoặc thuộc tính của nút/cạnh. Trong lab này, tổ chức, con người, chính sách, sản phẩm, địa điểm và công nghệ là các nút. Nhãn quan hệ phụ thuộc extractor: run LLM (`--extractor openai`, dùng cho báo cáo này) sinh predicate `UPPER_SNAKE_CASE` tự do theo ngữ cảnh câu — ví dụ thực tế `ANNOUNCED`, `IS_A_SUBSIDIARY_OF`, `DELIVER`, `REACHED`, `FORECAST`, `WAS`, `HAD`; còn extractor heuristic offline dùng tập nhãn cố định, bảo thủ như `REPORTS_FINANCIAL_RESULT`, `REPORTS_SALES`, `REPORTS_POLICY` và `MENTIONS`.

**Khử trùng lặp đồ thị.** Khử trùng lặp quan trọng vì những tên lặp như `Google`, `Alphabet` hoặc `Google LLC` có thể chia cắt một thực thể thật thành nhiều nút. Đồ thị bị phân mảnh làm giảm degree centrality, phá vỡ duyệt nhiều hop và có thể khiến bước trả lời bỏ sót bằng chứng.

**BFS so với tìm kiếm vector.** Tìm kiếm vector truy xuất độc lập các đoạn văn tương đồng về ngữ nghĩa. BFS mở rộng từ một thực thể đã khớp qua các cạnh tường minh, nên phù hợp hơn với câu hỏi nhiều hop cần kết nối công ty với chính sách, sản phẩm, đối tác, nhà cung cấp hoặc hệ quả.

## 2. Tóm tắt triển khai

- Tài liệu nguồn: 70 file văn bản trong `dataset/`.
- Triple ngữ nghĩa đã trích xuất: 331.
- Knowledge graph: 597 nút và 993 cạnh có hướng.
- Backend đồ thị: NetworkX `MultiDiGraph`.
- Flat RAG baseline: truy xuất TF-IDF trên toàn bộ tài liệu.
- GraphRAG: khớp thực thể, duyệt đồ thị tối đa 2 hop với bằng chứng truy nguyên được nguồn. Khi có thực thể khớp có cạnh, đường đi đồ thị dẫn dắt xếp hạng (graph 0.6 / lexical 0.4); không có thực thể thì lui về lexical thuần. Có thể bật tổng hợp câu trả lời bằng LLM qua `--answer-with-llm`.

## 3. Ảnh Knowledge Graph

![Knowledge graph](artifacts/knowledge_graph.png)

## 4. Tóm tắt benchmark

Benchmark gồm 20 câu hỏi **quan hệ / nhiều hop** được kiểm chứng thủ công: mỗi câu neo vào một thực thể có trong đồ thị nhưng được diễn đạt bằng từ vựng mà tài liệu đích **không lặp lại** (ví dụ hỏi "công ty mẹ", "thành viên chỉ số", "đối tác", "nhà quản lý đầu tư"). Đây là loại câu hỏi làm lộ rõ điểm yếu của truy xuất túi-từ và phát huy sức mạnh duyệt cạnh quan hệ của đồ thị. Mỗi câu có tài liệu đích, câu trả lời mong đợi, thuật ngữ bắt buộc và bằng chứng nguồn. Kết quả truy xuất:

| Hệ thống | Top-1 | Hit@3 | MRR | Độ bao phủ thuật ngữ trả lời |
| --- | ---: | ---: | ---: | ---: |
| Flat RAG (TF-IDF) | 0.30 | 0.45 | 0.457 | 0.40 |
| GraphRAG (entity + 2-hop) | 0.85 | 0.95 | 0.910 | 0.70 |

Theo thứ hạng tài liệu đích, **GraphRAG xếp cao hơn ở 13 câu, không thua câu nào, và hòa 7 câu**. Có **10 câu chỉ GraphRAG** đưa được tài liệu đích vào top 3, và **0 câu chỉ Flat RAG** làm được. Kết quả đầy đủ trong [artifacts/benchmark_20.csv](artifacts/benchmark_20.csv).

Cơ chế: khi câu hỏi khớp một thực thể có cạnh trong đồ thị, GraphRAG để các đường đi entity→quan hệ dẫn dắt việc xếp hạng (lexical TF-IDF chỉ làm fallback/phá hòa); nếu không có thực thể khớp, hệ thống lui về lexical thuần. Nhờ đó câu như B05 (Georgia, flat hạng 17 → graph hạng 1) hay B20 (lệnh cấm xe đốt trong của California, flat 13 → graph 1) được tìm đúng dù tài liệu đích gần như không dùng lại từ ngữ trong câu hỏi. Các câu hòa là những câu mà cả hai đã trả về tài liệu đích ở hạng cao (1–2). Giới hạn và diễn giải đầy đủ trong [artifacts/evaluation_analysis.md](artifacts/evaluation_analysis.md).

| ID | Câu hỏi | Hạng Flat RAG | Hạng GraphRAG | Hệ thống xếp cao hơn |
| --- | --- | ---: | ---: | --- |
| B01 | Tập đoàn mẹ nào sở hữu Cox Automotive, đơn vị đứng sau Chỉ số Tâm lý Đại lý? | 1 | 1 | Hòa |
| B02 | Hãng xe Trung Quốc nào vượt Tesla thành hãng bán xe điện số 1 cuối 2024? | 6 | 1 | GraphRAG |
| B03 | Ngoài Tesla, hai hãng xe lâu đời của Mỹ nào nằm trong chỉ số S&P 500? | 5 | 1 | GraphRAG |
| B04 | Văn phòng nào hợp tác với chính phủ để thúc đẩy dự án tiếp nhiên liệu/giao thông không phát thải? | 1 | 1 | Hòa |
| B05 | Bang Georgia thu hút bao nhiêu vốn đầu tư sản xuất EV, dẫn đầu toàn nước Mỹ? | 17 | 1 | GraphRAG |
| B06 | General Motors đầu tư bao nhiêu để ra mắt dải sản phẩm EV? | 6 | 2 | GraphRAG |
| B07 | Công ty nào là nhà quản lý đầu tư cho các quỹ KraneShares ETF? | 1 | 1 | Hòa |
| B08 | Năm 2022, quốc gia nào là thị trường lớn thứ hai của Tesla theo doanh số? | 2 | 2 | Hòa |
| B09 | Doanh số EV châu Âu đã vượt qua doanh số của ai lần đầu sau nhiều năm trong 2020? | 9 | 1 | GraphRAG |
| B10 | Ai trở thành bộ trưởng khoa học và công nghệ Trung Quốc năm 2007 và thúc đẩy ngành EV? | 4 | 1 | GraphRAG |
| B11 | Start-up sạc EV Numbat của Đức huy động bao nhiêu vốn series A? | 3 | 1 | GraphRAG |
| B12 | REE dự định mở Integration Center tại thành phố nào của Mỹ? | 1 | 1 | Hòa |
| B13 | Công ty nào dự đoán đầu 2023 rằng doanh số EV Mỹ sẽ vượt mốc một triệu? | 5 | 1 | GraphRAG |
| B14 | Trung Quốc công bố khoản đầu tư hạ tầng sạc bổ sung nào trong kế hoạch phục hồi COVID-19? | 6 | 5 | GraphRAG |
| B15 | Hiệp hội điện mặt trời nào được Washington Post vinh danh và đạt giải Best Nonprofit to Work For? | 2 | 1 | GraphRAG |
| B16 | Ai là tác giả chuyên mục dòng vốn ETF 'Flow & Tell' của iShares? | 1 | 1 | Hòa |
| B17 | Sáng kiến nào của Bộ Năng lượng sẽ cấp hơn 13 tỷ USD cải thiện độ tin cậy lưới điện Mỹ? | 4 | 1 | GraphRAG |
| B18 | Mẫu xe Polestar 3 ra mắt khi nào? | 1 | 1 | Hòa |
| B19 | EIA công khai mã nguồn mở của mình ở đâu? | 6 | 1 | GraphRAG |
| B20 | Cùng tín dụng thuế của Inflation Reduction Act, bang nào tuyên bố cấm bán xe đốt trong mới từ 2035? | 13 | 1 | GraphRAG |

## 5. Điểm yếu Flat RAG / Trường hợp GraphRAG có lợi thế

Flat RAG xếp hạng theo độ trùng từ vựng, nên thất bại khi câu hỏi và tài liệu đích mô tả cùng một quan hệ bằng từ ngữ khác nhau. Các trường hợp rõ nhất:

- **B05** ("Georgia thu hút bao nhiêu vốn đầu tư EV?"): tài liệu đích nói "Georgia continues to lead the states in EV investments ($31.2 billion)"; Flat RAG xếp hạng 17 vì nhiều tài liệu khác cũng đầy từ "investment/EV", còn GraphRAG đi thẳng cạnh `Georgia → LEADS_IN_EV_INVESTMENTS` lên hạng 1.
- **B20** ("bang nào cấm xe đốt trong từ 2035?"): Flat RAG hạng 13; GraphRAG neo vào `Inflation Reduction Act`/`California` và lên hạng 1.
- **B03** (thành viên S&P 500 ngoài Tesla), **B09** (châu Âu vượt doanh số của ai), **B02** (BYD vượt Tesla): đều là quan hệ giữa hai thực thể mà chỉ duyệt cạnh mới trả lời đúng — Flat RAG hạng 5–9, GraphRAG hạng 1.

Tổng cộng GraphRAG thắng 13, hòa 7, thua 0; không có câu nào Flat RAG xếp cao hơn. Đây là minh chứng có kiểm soát cho sức mạnh của truy xuất dựa trên đồ thị với câu hỏi quan hệ — khác với câu hỏi tra cứu factoid một-hop, nơi hai phương pháp thường ngang nhau.

Đánh giá này đo chất lượng truy xuất và độ bao phủ thuật ngữ của câu trả lời có căn cứ nguồn; nó không chấm độc lập hallucination của câu trả lời LLM tự do. Xem [artifacts/evaluation_analysis.md](artifacts/evaluation_analysis.md) để biết phương pháp và diễn giải chi tiết.

## 6. Phân tích chi phí và thời gian

- Model: `gpt-4o-mini`.
- Số lệnh gọi API: 70.
- Token indexing đã đo: 98.105 token (61.252 input; 36.853 output).
- Chi phí indexing ước tính: $0.03129960, với mức $0.15 / 1 triệu input token và $0.60 / 1 triệu output token.
- Thời gian trích xuất OpenAI đầy đủ: 800,173 giây.
- Thời gian dựng lại artifact từ checkpoint đã lưu: 0,488 giây.

Bước indexing một lần chiếm phần lớn chi phí và thời gian vì từng tài liệu nguồn được gửi đến extractor. Lần chạy này không lưu thời lượng benchmark ở query time, nên báo cáo không ước tính số liệu đó. Dữ liệu usage và mức giá đã ghi nằm trong [artifacts/token_usage.json](artifacts/token_usage.json); hồ sơ chạy đầy đủ nằm trong [artifacts/full_llm_indexing.md](artifacts/full_llm_indexing.md).

## 7. Khả năng tái lập

Cài dependency và chạy test:

```bash
make install
make test
```

Dựng lại artifact bằng heuristic extractor mặc định:

```bash
make run
```

Để dựng lại từ checkpoint trích xuất LLM, cấu hình API key và các biến giá, sau đó chạy:

```bash
OPENAI_API_KEY='...' \
OPENAI_INPUT_COST_PER_1M='0.15' \
OPENAI_OUTPUT_COST_PER_1M='0.60' \
uv run python graphrag_lab.py --extractor openai --model gpt-4o-mini
```

Extractor `openai` tái sử dụng các file thành công trong `artifacts/llm_cache/`; do đó một lần chạy đã hoàn tất có thể dựng lại artifact mà không lặp lại các lệnh gọi trích xuất.
