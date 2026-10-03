# K4-DAY17 — Memory Systems for AI Agent

Học viên: Nguyễn Hồng Thái — 2A202602894

So sánh hai agent trên cùng một benchmark tiếng Việt:

- `Baseline Agent`: chỉ nhớ trong cùng thread.
- `Advanced Agent`: short-term memory + `User.md` bền vững + compact memory khi hội thoại vượt ngưỡng.

Phân tích kết quả (bước 8) nằm trong [STEP8.md](STEP8.md).

## Cấu trúc

```
├── src/
│   ├── model_provider.py     # ProviderConfig, build_chat_model (openai, custom, gemini, anthropic, ollama, openrouter)
│   ├── config.py             # LabConfig, load_config
│   ├── memory_store.py       # estimate_tokens, UserProfileStore, extract_profile_updates, CompactMemoryManager
│   ├── offline_responder.py  # sinh câu trả lời offline, dùng chung cho hai agent
│   ├── agent_baseline.py     # BaselineAgent
│   ├── agent_advanced.py     # AdvancedAgent
│   ├── benchmark.py          # Standard + Long-Context Stress benchmark
│   └── test_agents.py        # 4 test
├── data/                     # input benchmark (giữ nguyên)
├── STEP8.md
└── README.md
```

## Chạy

Chế độ mặc định là offline và deterministic: không cần API key, không cần `.env`, không phụ thuộc thư mục `state/`. Chỉ cần Python >= 3.11 và `pytest`.

```bash
python src/benchmark.py
pytest src/test_agents.py -v
```

Lệnh benchmark in hai bảng **Standard Benchmark** và **Long-Context Stress Benchmark**, mỗi bảng đủ 6 cột: `Agent tokens only`, `Prompt tokens processed`, `Cross-session recall`, `Response quality`, `Memory growth (bytes)`, `Compactions`. Khi chạy, `User.md` được ghi vào `state/` (đã gitignore) và mỗi lần chạy benchmark đều xóa trạng thái cũ.

## Chế độ live (tùy chọn)

```bash
pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate
python src/benchmark.py --live
```

Tạo `.env` ở thư mục gốc:

```
LLM_PROVIDER=gemini            # openai | custom | gemini | anthropic | ollama | openrouter
LLM_MODEL=gemini-2.5-flash
GEMINI_API_KEY=...             # hoặc OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, CUSTOM_API_KEY + CUSTOM_BASE_URL, OLLAMA_BASE_URL
LLM_RPM=12                     # tùy chọn: giới hạn request/phút cho free tier
COMPACT_THRESHOLD_TOKENS=800
COMPACT_KEEP_MESSAGES=4
```

Ở chế độ live, agent dùng `create_agent` + `InMemorySaver`. Advanced có thêm tool đọc/ghi `User.md` (mỗi lượt ghi đều qua `validate_fact_write`), `dynamic_prompt` chèn hồ sơ người dùng và `SummarizationMiddleware`.
