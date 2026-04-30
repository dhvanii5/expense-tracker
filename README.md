# Expense Tracker

This repository contains the full expense tracker monorepo, including a FastAPI backend and a Vite + React frontend.

## Start Here

For app-specific setup, API usage, and deployment details, see:

- [backend/README.md](backend/README.md)
- [frontend/README.md](frontend/README.md)

## Repository Layout

- `backend/` - FastAPI API, transaction extraction logic, local vector store, and model integration
- `frontend/` - Vite React UI for chat-based expense and income entry
- `dataset/` - local dataset and cleanup scripts used during development
- `model/` - local model assets such as GGUF files

## Quick Backend Entry Points

- Direct app runner: `backend/main.py`
- Deployment wrapper: `backend/api_v2.py`

## Typical Commands

Backend dependency sync:

```bash
uv sync --directory backend
```

Frontend dependency install:

```bash
npm install --prefix frontend
```

Run backend directly:

```bash
uv run --directory backend .\main.py
```

Run backend with the deployment wrapper:

```bash
uv run --directory backend .\api_v2.py
```

Run frontend locally:

```bash
npm run dev --prefix frontend
```

The frontend dev server runs on `http://127.0.0.1:8080` and the backend runs on `http://127.0.0.1:9002` by default.

If you want the frontend to use a custom backend URL, set `VITE_API_BASE_URL` in the frontend environment.
