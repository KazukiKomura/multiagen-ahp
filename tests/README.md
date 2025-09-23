Tests for AI Chat endpoints

Overview
- Scenario test: Runs an end-to-end flow to prime a session and (optionally) send one chat turn.
- Load test: Simulates many concurrent users running the minimal flow needed for the chat to start.

Assumptions
- The Flask app is running locally: `http://127.0.0.1:5002`.
- A session is initialized via `GET /start/ai-facilitator`.
- Network access and API key are required if you include `/ai_chat` in the flow (it calls OpenAI). If not available, use `--skip-llm` to avoid hitting `/ai_chat`.

Quick Start
- Scenario test (without LLM):
  `python tests/scenario_test_ai_chat.py --skip-llm`

- Load test (50 concurrent, 1 chat turn, with LLM step skipped):
  `python tests/load_test_ai_chat.py --concurrency 50 --turns 0`

Options
- Both scripts accept `--host` and `--port` (default: 127.0.0.1:5002).
- `--turns` in load test controls number of `/ai_chat` turns per user (set 0 to skip LLM).

Note
- If you want to exercise `/ai_chat`, ensure the server has a valid `OPENAI_API_KEY` and outbound network.
- The scripts print basic latency metrics and error counts at the end.

