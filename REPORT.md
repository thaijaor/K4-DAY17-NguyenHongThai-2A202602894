# Day 17 — Memory Systems for AI Agent: Báo cáo

Học viên: Nguyễn Hồng Thái — 2A202602894

## Chạy lại

```bash
python -m venv .venv && .venv/Scripts/pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest
python src/benchmark.py          # offline, deterministic
python src/benchmark.py --live   # provider trong .env: LLM_PROVIDER, LLM_MODEL, <PROVIDER>_API_KEY, LLM_RPM (free tier)
pytest src/test_agents.py -v     # 9 tests
```

## Thiết kế

| Lớp memory | Nơi lưu | Baseline | Advanced |
|---|---|---|---|
| Short-term | message trong `thread_id` | có (giữ toàn bộ) | có (giữ `COMPACT_KEEP_MESSAGES` message gần nhất) |
| Persistent | `state/profiles/<user>/User.md` | không | có |
| Compact | summary trong `CompactMemoryManager` | không | có, kích hoạt khi thread > `COMPACT_THRESHOLD_TOKENS` (800) |

- Mỗi lượt Advanced: `extract_profile_updates()` → `upsert_fact()` vào `User.md` → append vào compact memory → prompt = system + `User.md` + summary + message gần nhất.
- Offline: cả hai agent dùng chung `offline_responder.compose_reply()`, chỉ khác nguồn fact (Baseline: thread hiện tại; Advanced: `User.md`), nên so sánh công bằng.
- Live: `create_agent` + `InMemorySaver`; Advanced thêm tool `read_user_memory`/`save_user_fact`, `dynamic_prompt` chèn `User.md`, `SummarizationMiddleware`. Hỗ trợ `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`.
- Đếm token: `Agent tokens only` = message + reply (+ với Advanced: token ghi `User.md` và token summary, vì đây là output của model trong hệ thống thật). `Prompt tokens processed` = tổng ngữ cảnh gửi vào model qua mọi lượt.

## Kết quả (offline)

**Standard Benchmark** — 10 hội thoại, 101 lượt, 14 câu recall ở thread mới

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 2418 | 15065 | 0.00 | 0.00 | 0 | 0 |
| Advanced | 2742 | 26063 | 1.00 | 1.00 | 305 | 0 |

**Long-Context Stress Benchmark** — 1 hội thoại, 16 lượt dài, 3 câu recall

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 2588 | 22170 | 0.00 | 0.00 | 0 | 0 |
| Advanced | 3089 | 11045 | 1.00 | 1.00 | 241 | 4 |

## Kết quả (live, `gemini-3.5-flash-lite`, `LLM_RPM=12`)

Prompt tokens lấy từ `usage_metadata` của provider. Compactions đếm bằng `CompactMemoryManager` chạy song song với `SummarizationMiddleware`.

| Suite | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality (LLM judge) | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|---|
| Standard | Baseline | 6915 | 35067 | 0.11 | 0.23 | 0 | 0 |
| Standard | Advanced | 22362 | 90191 | 0.96 | 0.68 | 398 | 40 |
| Stress | Baseline | 4335 | 37609 | 0.00 | 0.00 | 0 | 0 |
| Stress | Advanced | 6956 | 17999 | 1.00 | 0.93 | 336 | 17 |

- Xu hướng giống bản offline: Standard thì Advanced tốn hơn (prompt +157%); Stress thì Advanced giảm 52% prompt tokens.
- Câu trả lời của LLM dài hơn bản offline nên thread chạm ngưỡng 800 token ngay ở bộ Standard (40 lần compact). Phần token để sinh summary và gọi tool cũng làm agent tokens tăng mạnh (+223%).
- Baseline có recall 0.11 vì câu trả lời tình cờ chứa cụm "ngắn gọn", không phải vì nhớ được.
- Lỗi lộ ra khi chạy live: LLM gọi `save_user_fact` để ghi chủ đề tin tức vào `interests` ("Energy policy", "so sánh CAPEX...") và chú thích tự bịa vào `pet` ("corgi tên Bơ (thường xuyên dùng làm ví dụ test case)"). Đã sửa bằng `validate_fact_write()` (xem mục Bonus). Bảng trên là số liệu chạy trước khi sửa; chưa chạy live lại được vì key free tier đã hết quota 500 request/ngày.

## Phân tích

- **Recall:** Baseline chỉ có `SessionState` theo thread nên recall ở thread mới bằng 0. Advanced đọc `User.md` nên trả lời đúng cả fact đã đính chính (Huế, MLOps engineer ở standard; Đà Nẵng ở stress).
- **Hội thoại ngắn, Advanced tốn hơn:** thread standard chỉ khoảng 450 token, chưa chạm ngưỡng nên không compact. `User.md` (~80 token) cùng system prompt dài hơn bị gửi lại ở mọi lượt, cộng thêm token ghi memory. Kết quả: prompt tokens +73%, agent tokens +13%.
- **Hội thoại dài, compact thắng:** prompt của Baseline tăng bậc hai theo số lượt, vì lượt thứ k gửi lại cả k message. Advanced compact 4 lần, ngữ cảnh mỗi lượt bị chặn ở mức khoảng ngưỡng + `User.md` + summary, nên prompt tokens giảm 50%. Compact chủ yếu tối ưu `prompt tokens processed`, còn `agent tokens only` vẫn tăng khoảng 19% do phải sinh summary và ghi memory.
- **Memory growth:** `User.md` tăng theo số loại fact, không tăng theo số lượt: 305 byte sau 101 lượt, vì fact bị ghi đè thay vì nối thêm. Rủi ro: (1) lưu sai fact thì sai ở mọi phiên sau; (2) nếu schema mở thì file phình và tốn prompt ở mọi lượt; (3) summary heuristic có thể mất chi tiết. Fact quan trọng đã được rút vào `User.md` trước khi compact nên không phụ thuộc vào summary.

## Bonus

| Bonus | Cách làm | Lợi ích | Rủi ro |
|---|---|---|---|
| Confidence threshold | Mỗi fact có confidence; mệnh đề giả định (`nếu`) = 0.4, dưới ngưỡng 0.6 thì không ghi | Tránh lưu "Nếu sau này mình nhắc Đà Nẵng..." | Ngưỡng cứng có thể bỏ sót fact thật nói kiểu giả định |
| Tránh lưu câu hỏi | Câu hỏi hoặc yêu cầu nhắc lại (`?`, `gì`, `nhớ lại xem`) bị bỏ qua khi trích fact | "đồ uống yêu thích là gì" không bị lưu thành fact | Câu khẳng định có chữ "gì" có thể bị bỏ qua |
| Grounding guardrail cho tool ghi của LLM | `validate_fact_write()`: value phải xuất hiện nguyên văn trong lời người dùng ở thread hiện tại, và mệnh đề chứa nó phải vượt confidence threshold (loại câu hỏi, câu đùa, câu giả định, ngữ cảnh "tạm thời/hôm nay"); `interests` chỉ nhận từ danh sách kỹ thuật; value > 60 ký tự hoặc có chú thích bị từ chối | Chặn đúng các giá trị sai Gemini đã ghi khi chạy live (có test) | Chặn cả fact đúng khi LLM diễn đạt lại thay vì chép nguyên văn; chỉ đối chiếu với các message còn giữ sau compact |
| Conflict handling | `upsert_fact` ghi đè; mệnh đề có `không còn`, `chỉ là`, `đùa`, `họp`, `lúc đầu` bị loại | Không giữ song song Huế/Đà Nẵng hay backend/MLOps; bỏ qua nhiễu Hà Nội, product manager | Danh sách cue viết tay, khó tổng quát sang cách diễn đạt khác |
| Entity extraction | `response_style` tách slot `length/format/examples/emphasis`, merge theo slot; format mơ hồ không đè "3 bullet" | Giữ "3 bullet" dù về sau user chỉ nói "có cấu trúc" | Thêm logic merge phải test |
| Memory decay | `interests` xếp theo lần nhắc gần nhất, giữ tối đa 6 | Chặn file phình theo thời gian | Sở thích cũ nhưng vẫn đúng có thể bị đẩy ra |

Giới hạn: phần trích fact offline là regex với danh sách thành phố và nghề cố định, chỉ phù hợp phạm vi dataset. Ở chế độ live, LLM có thể ghi thêm fact qua tool `save_user_fact`, nhưng mọi lượt ghi đều phải qua `validate_fact_write()`.
