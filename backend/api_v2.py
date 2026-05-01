"""Standalone deployment API exposing only frontend-used endpoints."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import main

app = FastAPI(lifespan=main.lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(req: main.ChatRequest):
    return await main.chat(req)


@app.post("/save")
async def save(req: main.SaveRequest):
    return await main.save(req)


@app.post("/analytics")
async def analytics(req: main.AnalyticsRequest):
    return await main.analytics(req)


@app.get("/transactions")
async def transactions():
    return await main.transactions()


@app.delete("/transactions/{tx_id}")
async def delete_transaction(tx_id: str):
    return await main.delete_transaction(tx_id)


@app.post("/debug")
async def debug(req: main.ChatRequest):
    return await main.debug(req)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "api_v2",
        "model": main.LLAMA_SERVER_MODEL or None,
        "llama_server_url": main.LLAMA_SERVER_URL or None,
    }


if __name__ == "__main__":
    import os
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9002"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    uvicorn.run("api_v2:app", host=host, port=port, reload=reload)
