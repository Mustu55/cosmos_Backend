"""My COSMOS API.
Local:   uvicorn app.main:app --reload --port 8020
Server:  uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import assistant, graph, opportunities, papers_api, recommend, search, trends

DEFAULT_ORIGINS = "http://localhost:5173,http://localhost:5174,http://localhost:4173"
ORIGINS = [
    o.strip().rstrip("/")
    for o in os.getenv("COSMOS_CORS_ORIGINS", DEFAULT_ORIGINS).split(",")
    if o.strip()
]

app = FastAPI(title="My COSMOS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(papers_api.router)
app.include_router(search.router)
app.include_router(recommend.router)
app.include_router(graph.router)
app.include_router(trends.router)
app.include_router(opportunities.router)
app.include_router(assistant.router)


@app.get("/")
def root():
    """Service entry point; API documentation is available at /docs."""
    return {
        "status": "ok",
        "service": "My COSMOS API",
        "health": "/health",
        "docs": "/docs",
        "answer_provider": assistant.PROVIDER.name,
    }


@app.get("/health")
def health():
    """Hosts call this to check the API is alive."""
    return {"status": "ok", "answer_provider": assistant.PROVIDER.name}
