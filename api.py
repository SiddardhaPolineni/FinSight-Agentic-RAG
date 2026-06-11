"""
FinSight FastAPI Backend
─────────────────────────
Routes:
  POST   /FinSight/stream        — streaming chat (SSE, token-by-token)
  POST   /FinSight/chat          — non-streaming chat (full JSON response)
  GET    /FinSight/history/{id}  — retrieve session history
  DELETE /FinSight/history/{id}  — clear session history
"""

import json
import logging
import uuid
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage

import config as cfg
from src.graph import graph
from src.retriever import load_bm25_index
from src.schemas import (
    ChatRequest,
    ChatResponse,
    HistoryMessage,
    HistoryResponse,
    SourceDoc,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Session store ─────────────────────────────────────────────────────────────
_sessions: dict[str, ChatMessageHistory] = {}


def _get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _sessions:
        _sessions[session_id] = ChatMessageHistory()
    return _sessions[session_id]


def _build_initial_state(question: str, history: ChatMessageHistory) -> dict:
    return {
        "messages":           history.messages,
        "question":           question,
        "rephrased_query":    None,
        "intent":             None,
        "companies":          [],
        "years":              [],
        "metrics":            [],
        "statement_type":     None,
        "chart_type":         None,
        "csv_result":         None,
        "csv_dataframe_json": None,
        "retrieved_docs":     [],
        "rag_answer":         None,
        "chart_spec":         None,
        "answer":             None,
        "error":              None,
    }


def _build_sources(final_state: dict) -> list[SourceDoc]:
    return [
        SourceDoc(
            source=d.get("source", ""),
            company=d.get("company", "").title(),
            year=d.get("year", ""),
            score=round(d.get("score", 0.0), 4),
            snippet=d.get("content", "")[:250],
        )
        for d in (final_state.get("retrieved_docs") or [])
    ]


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinSight Financial Research Assistant",
    description="Agentic RAG over SEC 10-K filings and financial CSV data.",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    n = load_bm25_index()
    logger.info("FinSight API ready. BM25 docs: %d", n)


# ── Streaming endpoint ────────────────────────────────────────────────────────

async def _sse_stream(answer: str) -> AsyncIterator[str]:
    """Yield answer word-by-word as SSE data events."""
    words = answer.split(" ")
    for i, word in enumerate(words):
        token = word if i == len(words) - 1 else word + " "
        yield f"data: {json.dumps({'token': token})}\n\n"


@app.post("/FinSight/stream")
async def stream_chat(req: ChatRequest):
    """
    SSE streaming endpoint.
    Streams the answer token-by-token, then sends a final [DONE] event
    containing chart_spec and session_id.
    """
    session_id = req.session_id or str(uuid.uuid4())
    history    = _get_history(session_id)

    logger.info("[%s] Stream: %s", session_id, req.question)

    initial_state = _build_initial_state(req.question, history)
    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as e:
        logger.error("Graph error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    answer     = final_state.get("answer") or "No answer generated."
    chart_spec = final_state.get("chart_spec")
    sources    = [s.model_dump() for s in _build_sources(final_state)]

    history.add_user_message(req.question)
    history.add_ai_message(answer)

    async def event_generator():
        async for event in _sse_stream(answer):
            yield event
        done_payload = json.dumps({
            "session_id": session_id,
            "chart_spec": chart_spec,
            "sources":    sources,
        })
        yield f"data: [DONE]{done_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Non-streaming endpoint ────────────────────────────────────────────────────

@app.post("/FinSight/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Full response in one JSON payload — useful for testing."""
    session_id = req.session_id or str(uuid.uuid4())
    history    = _get_history(session_id)

    logger.info("[%s] Chat: %s", session_id, req.question)

    initial_state = _build_initial_state(req.question, history)
    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as e:
        logger.error("Graph error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    answer = final_state.get("answer") or "No answer generated."
    history.add_user_message(req.question)
    history.add_ai_message(answer)

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        chart_spec=final_state.get("chart_spec"),
        sources=_build_sources(final_state),
    )


# ── History endpoints ─────────────────────────────────────────────────────────

@app.get("/FinSight/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    history = _get_history(session_id)
    msgs = [
        HistoryMessage(
            role="user" if isinstance(m, HumanMessage) else "assistant",
            content=m.content,
        )
        for m in history.messages
    ]
    return HistoryResponse(session_id=session_id, messages=msgs)


@app.delete("/FinSight/history/{session_id}")
async def clear_history(session_id: str):
    _sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}
