# Expense Tracker

This repository contains the full expense tracker monorepo, including a FastAPI backend and a Vite + React frontend.

## Developer Quickstart (Docker)

Use this path when another developer clones the repo and needs to run and test quickly.

### 1) Clone and prepare environment

```bash
git clone <your-repo-url>
cd expense-tracker
```

Linux/macOS:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

### 2) Configure model access

Set `LLAMA_SERVER_URL` (and optionally completions/embeddings URLs) in `.env`.

- If you have a shared remote model server, the developer does **not** need to install `llama.cpp`.
- If no shared model server exists, they must run `llama-server` locally and point `.env` to it.

Recommended `.env` values:

```dotenv
BACKEND_PORT=9002
FRONTEND_PORT=8080
VITE_API_BASE_URL=http://localhost:9002

LLAMA_SERVER_URL=http://host-or-ip:8080
LLAMA_SERVER_COMPLETIONS_URL=http://host-or-ip:8080/completion
LLAMA_SERVER_EMBEDDINGS_URL=http://host-or-ip:8080/embedding
LLAMA_SERVER_MODEL=qwen3b-dhvani-q8_0.gguf
```

Model access options:

- Shared remote model server (recommended for team onboarding): no local `llama.cpp` required.
- Local model server per developer: requires local `llama-server` + GGUF model on that machine.

### 3) Start backend + frontend

```bash
docker compose up --build -d
```

### 4) Verify services

- Frontend: `http://localhost:8080`
- Backend health: `http://localhost:9002/health`

The health response should show backend mode:

- `llama_server_remote` when model URL is configured and reachable
- `heuristic_fallback` when model server is unavailable

### 5) Stop services

```bash
docker compose down
```

## Quick Troubleshooting

- `backend` is `heuristic_fallback` in `/health`:
  - Check `LLAMA_SERVER_URL` and optional completion/embedding URLs in `.env`.
  - Confirm the model server is reachable from Docker host/network.
- Frontend loads but API calls fail:
  - Ensure `VITE_API_BASE_URL` points to backend host/port exposed by compose.
  - Rebuild after changing frontend env: `docker compose up --build -d`.
- Port already in use:
  - Change `BACKEND_PORT` or `FRONTEND_PORT` in `.env`, then restart compose.

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
