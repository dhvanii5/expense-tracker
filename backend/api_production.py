"""Production deployment API - Port 7000 hardcoded for remote server."""

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
        "service": "api_production:7000",
        "model": main.MODEL,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_production:app", host="0.0.0.0", port=7000, reload=False)
