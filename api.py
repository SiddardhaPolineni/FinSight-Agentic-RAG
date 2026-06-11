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
from typing import Optional

import numpy as np
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


# ── Numpy-safe JSON encoder ───────────────────────────────────────────────────

class _SafeEncoder(json.JSONEncoder):
    """Converts numpy scalars/arrays to native Python so json.dumps never crashes."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _dumps(obj) -> str:
    return json.dumps(obj, cls=_SafeEncoder)

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
    import asyncio
    from src.retriever.schema_retriever import SchemaRetriever

    n = load_bm25_index()
    logger.info("FinSight API ready. BM25 docs: %d", n)

    # Pre-warm the schema cache for all 3 statement types.
    # Each call costs ~1.5 s (embed + Pinecone). Running them concurrently
    # means startup adds ~1.5 s total instead of 4.5 s, and every subsequent
    # data query skips the schema fetch entirely (0 ms from cache).
    _sr = SchemaRetriever()
    warmup_questions = {
        "income_statement": "revenue net income gross profit operating income ebitda eps margin",
        "balance_sheet":    "total assets liabilities equity debt cash goodwill working capital",
        "cash_flow":        "free cash flow operating cash capex capital expenditure investing financing",
    }

    async def _warm(stmt: str, q: str):
        try:
            await asyncio.to_thread(_sr.get_schema, q, stmt, 10)
            logger.info("Schema cache warmed: %s", stmt)
        except Exception as e:
            logger.warning("Schema warmup failed for %s: %s", stmt, e)

    await asyncio.gather(*[_warm(s, q) for s, q in warmup_questions.items()])
    logger.info("Schema cache pre-warm complete")


# ── Streaming endpoint ────────────────────────────────────────────────────────


@app.post("/FinSight/stream")
async def stream_chat(req: ChatRequest):
    """
    SSE streaming endpoint.
    Streams the answer token-by-token, then sends a final [DONE] event
    containing session_id. chart_spec is sent as a separate event to
    avoid oversized SSE lines that cause chunked-read errors.
    """
    session_id = req.session_id or str(uuid.uuid4())
    history    = _get_history(session_id)

    logger.info("[%s] Stream: %s", session_id, req.question)

    initial_state = _build_initial_state(req.question, history)

    async def event_generator():
        # ── Phase 1: run the graph, send keepalive pings every 5 s ──────────
        import asyncio

        result_holder: dict = {}
        error_holder:  dict = {}

        async def run_graph():
            try:
                result_holder["state"] = await graph.ainvoke(initial_state)
            except Exception as exc:
                error_holder["exc"] = exc

        task = asyncio.create_task(run_graph())

        # Send a comment-line keepalive while the graph is running so the
        # client connection stays alive (httpx/nginx won't close idle streams)
        while not task.done():
            yield ": keepalive\n\n"
            await asyncio.sleep(5)

        await task  # ensure any exception is surfaced

        if "exc" in error_holder:
            err_msg = f"Pipeline error: {error_holder['exc']}"
            logger.error(err_msg, exc_info=error_holder["exc"])
            yield f"data: {_dumps({'token': err_msg})}\n\n"
            yield f"data: [DONE]{_dumps({'session_id': session_id, 'has_chart': False, 'sources': []})}\n\n"
            return

        final_state = result_holder["state"]
        answer      = final_state.get("answer") or "No answer generated."
        chart_spec  = final_state.get("chart_spec")
        sources     = [s.model_dump() for s in _build_sources(final_state)]

        history.add_user_message(req.question)
        history.add_ai_message(answer)

        # ── Phase 2: stream tokens word-by-word ──────────────────────────────
        words = answer.split(" ")
        for i, word in enumerate(words):
            token = word if i == len(words) - 1 else word + " "
            yield f"data: {_dumps({'token': token})}\n\n"

        # ── Phase 3: send chart_spec as its own chunked events ───────────────
        if chart_spec:
            chart_json  = _dumps(chart_spec)
            chunk_size  = 16 * 1024
            total_parts = (len(chart_json) + chunk_size - 1) // chunk_size
            for idx in range(total_parts):
                part = chart_json[idx * chunk_size : (idx + 1) * chunk_size]
                yield f"data: {_dumps({'chart_part': part, 'part_idx': idx, 'total_parts': total_parts})}\n\n"

        # ── Phase 4: terminal [DONE] event ────────────────────────────────────
        done_payload = _dumps({
            "session_id": session_id,
            "has_chart":  chart_spec is not None,
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
