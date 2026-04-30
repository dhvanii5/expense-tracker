# Expense Tracker Backend API

This folder contains the FastAPI backend for the expense tracker. It exposes JSON endpoints for chat-based transaction entry, saving records, analytics, transaction listing, deletion, debugging, and health checks.

## Which app should I run?

Use `main.py` for direct local runs. It now starts `uvicorn` when executed directly.

Use `api_v2.py` for deployment or standalone hosting. It is a thin wrapper around the same backend logic.

Install and sync the backend environment with `uv`:

```bash
uv sync --directory backend
```

`llama-cpp-python` is not installed by default. On Windows, `uv` may fall back to a source build, which requires a native C/C++ toolchain such as Visual Studio Build Tools.

If you want to enable the local GGUF model path in `main.py`, install the optional extra with a supported environment:

```bash
uv sync --directory backend --extra local-llm
```

Run the backend directly:

```bash
uv run --directory backend .\main.py
```

Run the deployment wrapper:

```bash
uv run --directory backend .\api_v2.py
```

If you prefer explicit `uvicorn` commands during development:

```bash
uv run --directory backend uvicorn main:app --reload --host 0.0.0.0 --port 9002
```

## Base URL

If you run the server locally on port 9002, the base URL is:

```text
http://127.0.0.1:9002
```

If you use a different host or port, replace it in the examples below.

## Common Notes

- All endpoints accept and return JSON.
- CORS is enabled for browser clients.
- `chat` is the main endpoint used by the frontend.
- `health` is useful for checking that the server and local model are loaded.
- Date/time parsing is handled in `time_parser.py` inside this folder.
- Local Chroma data is stored under `backend/finance_memory/`.
- The local model file path defaults to `../model/qwen3b-dhvani-q8_0.gguf`.

## Endpoints

### `POST /chat`

Main conversation endpoint. Send a user message and optional session data if you are continuing a follow-up question.

Example expense input:

```json
{
  "message": "spent 599 for movie tickets at pvr 3days back"
}
```

Example follow-up response:

```json
{
  "message": "UPI",
  "session_data": {
    "intent": "expense",
    "items": [
      {
        "amount": 599,
        "category": "Entertainment",
        "currency": "INR",
        "item": "Movie Tickets",
        "merchant": "PVR",
        "payment_method": null,
        "remarks": "Paid for Movie Tickets at PVR.",
        "datetime": "2026-04-26T00:00:00",
        "bill_no": null,
        "source": null,
        "payer": null
      }
    ]
  },
  "followup_field": "payment_method"
}
```

Typical `chat` response when a field is still missing:

```json
{
  "status": "followup",
  "question": "How did you pay at PVR? (e.g., cash, UPI, card)",
  "followup_field": "payment_method",
  "is_optional_followup": true
}
```

### `POST /save`

Save a fully formed transaction entry.

Example:

```json
{
  "entry": {
    "intent": "expense",
    "items": [
      {
        "amount": 599,
        "category": "Entertainment",
        "currency": "INR",
        "item": "Movie Tickets",
        "merchant": "PVR",
        "payment_method": "UPI",
        "remarks": "Paid for Movie Tickets at PVR.",
        "datetime": "2026-04-26T00:00:00",
        "bill_no": null,
        "source": null,
        "payer": null
      }
    ]
  }
}
```

### `POST /analytics`

Get totals and breakdowns like spending, income, balance, and category breakdown.

Example:

```json
{
  "query_type": "total_expense",
  "filters": {
    "time_range": "this_month",
    "category": null,
    "start_date": null,
    "end_date": null
  }
}
```

Supported `query_type` values:

- `total_expense`
- `total_income`
- `balance`
- `category_breakdown`

Supported `time_range` values:

- `today`
- `this_week`
- `this_month`
- `last_month`
- `custom`
- `all_time`

### `GET /transactions`

Return all stored transactions.

Example:

```bash
curl http://127.0.0.1:9002/transactions
```

### `DELETE /transactions/{tx_id}`

Delete one transaction by id.

Example:

```bash
curl -X DELETE http://127.0.0.1:9002/transactions/3b6a3b52-6cc0-4d25-8d3e-7f7c2a2f0a11
```

### `POST /debug`

Debug endpoint that returns extraction details, retrieval context, and model output metadata.

Example:

```json
{
  "message": "paid 250 for groceries today"
}
```

### `GET /health`

Check if the backend is alive and whether the model is loaded.

Example response:

```json
{
  "status": "ok",
  "service": "api_v2",
  "model": "qwen3b-dhvani-q8_0.gguf",
  "model_path": "D:\\SLM\\expense-tracker\\model\\qwen3b-dhvani-q8_0.gguf"
}
```

## Quick curl Examples

Chat:

```bash
curl -X POST http://127.0.0.1:9002/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"spent 250 on groceries today\"}"
```

Save:

```bash
curl -X POST http://127.0.0.1:9002/save ^
  -H "Content-Type: application/json" ^
  -d "{\"entry\":{\"intent\":\"expense\",\"items\":[{\"amount\":250,\"category\":\"Groceries\",\"currency\":\"INR\",\"item\":\"Groceries\",\"merchant\":\"DMart\",\"payment_method\":\"Cash\",\"remarks\":\"Paid for groceries at DMart.\",\"datetime\":\"2026-04-29T00:00:00\",\"bill_no\":null,\"source\":null,\"payer\":null}]}}"
```

Analytics:

```bash
curl -X POST http://127.0.0.1:9002/analytics ^
  -H "Content-Type: application/json" ^
  -d "{\"query_type\":\"balance\",\"filters\":{\"time_range\":\"this_month\",\"category\":null,\"start_date\":null,\"end_date\":null}}"
```

## Tips for Developers

- Sync dependencies with `uv sync --directory backend`.
- Run tests with `uv run --directory backend pytest`.
- You can start the backend with `uv run --directory backend .\main.py`.
- If you are integrating from a frontend, keep the `session_data` from `chat` responses and send it back when answering follow-up questions.
- The backend can infer dates like `today`, `3 days back`, `last Monday`, and explicit calendar dates.
- Use `/debug` when a message is not behaving as expected; it shows the internal extraction and retrieval state.

## Files Related to the API

- `main.py` - core backend logic and routes
- `api_v2.py` - deployment wrapper for the backend
- `time_parser.py` - relative date and bill number parsing helpers
