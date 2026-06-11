"""
FinSight — Streamlit UI
────────────────────────
Clean chat interface:
  • Streams the answer token-by-token via SSE
  • Renders Plotly charts when the response includes one
  • No intent / metadata shown to the user
"""

import json
import uuid

import httpx
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight - Financial Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000/FinSight"

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id"   not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # [{"role", "content", "chart_spec?"}]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 FinSight")
    st.caption("AI-powered financial research")

    st.divider()
    st.subheader("📊 Data Coverage")
    st.markdown("""
**Financial Statements**
- Apple · Google · Microsoft · Nvidia
- Income Statement · Balance Sheet · Cash Flow
- Fiscal Years 2021 – 2025

**SEC 10-K Filings**
- Google · Microsoft · NVIDIA
- Years: 2023 · 2024 · 2025
    """)

    st.divider()
    st.subheader("💡 Try asking")
    examples = [
        "What was Nvidia's revenue from 2022 to 2025?",
        "Compare Apple and Microsoft net income for 2024",
        "What are Google's key risk factors in its 2024 10-K?",
        "Plot Nvidia's free cash flow from 2021 to 2025",
        "Show a bar chart of revenue for all companies in 2024",
        "What did Microsoft say about AI strategy in 2024?",
        "Apple's debt-to-equity ratio trend",
        "Compare operating margins of Google vs Microsoft",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:25]}"):
            st.session_state["prefill"] = ex

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        try:
            httpx.delete(f"{API_BASE}/history/{st.session_state.session_id}", timeout=5)
            st.session_state.session_id = str(uuid.uuid4())
        except Exception:
            pass
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("FinSight - Financial Research Assistant")
st.caption("Ask about financial statements, SEC filings, or request charts.")
st.divider()

# ── Render existing chat history ──────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chart_spec"):
            st.plotly_chart(
                go.Figure(msg["chart_spec"]),
                use_container_width=True,
                key=f"hist_chart_{id(msg)}",
            )


# ── Streaming helpers ─────────────────────────────────────────────────────────

def stream_from_api(question: str, session_id: str):
    """
    Generator that calls the SSE /stream endpoint and yields:
      - str tokens as they arrive
      - a dict {"chart_spec": ..., "session_id": ...} as the final item
    """
    with httpx.stream("POST",f"{API_BASE}/stream",json={"question": question, "session_id": session_id},timeout=120.0) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload.startswith("[DONE]"):
                # Final event — contains chart_spec and session_id
                meta_json = payload[len("[DONE]"):]
                try:
                    yield json.loads(meta_json)
                except Exception:
                    yield {}
                return
            try:
                chunk = json.loads(payload)
                yield chunk.get("token", "")
            except Exception:
                continue


# ── Chat input ────────────────────────────────────────────────────────────────
prefill    = st.session_state.pop("prefill", "")
user_input = st.chat_input("Ask a financial question…")

if prefill and not user_input:
    user_input = prefill

if user_input:
    # Show user message immediately
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream assistant response
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        chart_placeholder  = st.empty()

        full_answer = ""
        chart_spec  = None

        try:
            for chunk in stream_from_api(user_input, st.session_state.session_id):
                if isinstance(chunk, str):
                    # Token chunk — append and re-render
                    full_answer += chunk
                    answer_placeholder.markdown(full_answer + "▌")   # typing cursor
                elif isinstance(chunk, dict):
                    # Final DONE event
                    chart_spec = chunk.get("chart_spec")
                    # Update session_id if backend rotated it
                    if chunk.get("session_id"):
                        st.session_state.session_id = chunk["session_id"]

            # Remove cursor, show final answer
            answer_placeholder.markdown(full_answer)

            # Render chart if present
            if chart_spec:
                with chart_placeholder:
                    st.plotly_chart(
                        go.Figure(chart_spec),
                        use_container_width=True,
                        key=f"new_chart_{uuid.uuid4()}",
                    )

        except httpx.ConnectError:
            full_answer = (
                "⚠️ Cannot connect to the FinSight API. "
                "Make sure the backend is running:\n"
                "```\nuvicorn src.api:app --reload --port 8000\n```"
            )
            answer_placeholder.error(full_answer)

        except Exception as e:
            full_answer = f"⚠️ Error: {e}"
            answer_placeholder.error(full_answer)

        # Persist to chat history
        st.session_state.chat_history.append({
            "role":       "assistant",
            "content":    full_answer,
            "chart_spec": chart_spec,
        })

    st.rerun()
