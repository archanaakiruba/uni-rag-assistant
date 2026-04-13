"""
ABC RAG Assistant — FastAPI application.

Endpoints:
  GET  /           → serves static/index.html (chat UI)
  POST /ask        → { user_id, question } -> { answer, sources }
  GET  /health

Run with:
  uvicorn app:app --reload
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from src.models import AskRequest, AskResponse
from src.state import ConversationState
from src import intent_parser
from src.retriever import HybridRetriever
from src.context_builder import ContextBuilder
from src.prompt_builder import PromptBuilder
from src.generator import LLMGenerator
from src import guardrails

if not config.OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
    )

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ABC RAG Assistant",
    description="Answers prospective student questions about ABC University.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (chat UI)
_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# ---------------------------------------------------------------------------
# Singletons (initialised once at startup)
# ---------------------------------------------------------------------------

state = ConversationState()
retriever = HybridRetriever()
context_builder = ContextBuilder()
prompt_builder = PromptBuilder()
generator = LLMGenerator()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/")
async def serve_ui() -> FileResponse:
    index_path = os.path.join(_static_dir, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Chat UI not found.")
    return FileResponse(index_path)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    # 1. Load session
    session = state.get_session(request.user_id)

    # 2. Parse intent (async — LLM fallback if rule-based returns UNKNOWN)
    intent = await intent_parser.parse(request.question, session)

    # 3. Enrich session profile
    state.enrich_profile(session, intent)

    # 4. Guardrail: out of scope (fast check before expensive retrieval)
    if guardrails.is_out_of_scope(intent):
        return AskResponse(answer=config.SCOPE_RESPONSE, sources=[])

    # 5. Retrieve (async — ChromaDB + BM25 offloaded to executor)
    evidence = await retriever.retrieve(intent)

    # 6. Guardrail: confidence check
    if guardrails.confidence_too_low(evidence, intent):
        return AskResponse(answer=config.FALLBACK_RESPONSE, sources=[])

    # 7. Build context
    context_str, candidate_sources = context_builder.build(evidence)

    # 8. Build prompt
    messages = prompt_builder.build(request.question, context_str, session)

    # 9. Generate (async — GPT-4o call)
    answer, sources = await generator.generate(messages, candidate_sources)

    # 10. Save turn
    state.add_turn(request.user_id, request.question, answer, intent)

    return AskResponse(answer=answer, sources=sources)
