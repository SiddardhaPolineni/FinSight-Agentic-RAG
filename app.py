"""
FinSight — Streamlit UI
────────────────────────
Clean chat interface:
  • Streams the answer token-by-token via SSE
  • Renders Plotly charts when the response includes one
  • No intent / metadata shown to the user
"""

import json
import random
import time
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

# ── Thinking animation CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@keyframes thinking-pulse {
    0%   { opacity: 1; }
    50%  { opacity: 0.35; }
    100% { opacity: 1; }
}
@keyframes dot-bounce {
    0%, 80%, 100% { transform: translateY(0); }
    40%            { transform: translateY(-6px); }
}
.thinking-container {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 4px;
    animation: thinking-pulse 2s ease-in-out infinite;
}
.thinking-dots {
    display: flex;
    gap: 4px;
}
.thinking-dots span {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #7c83fd;
    display: inline-block;
    animation: dot-bounce 1.2s ease-in-out infinite;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.15s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.30s; }
.thinking-text {
    color: #9ca3af;
    font-style: italic;
    font-size: 0.92em;
}
</style>
""", unsafe_allow_html=True)

THINKING_MESSAGES = [
    "Searching financial records",
    "Analyzing statements",
    "Crunching the numbers",
    "Scanning SEC filings",
    "Synthesizing insights",
    "Running calculations",
    "Fetching market data",
    "Thinking it through",
    "Consulting the data",
    "Preparing your answer",
]

def thinking_html(msg: str) -> str:
    return (
        '<div class="thinking-container">'
        '  <div class="thinking-dots">'
        '    <span></span><span></span><span></span>'
        '  </div>'
        f' <span class="thinking-text">{msg}…</span>'
        '</div>'
    )


def _escape_dollars(text: str) -> str:
    """Escape $ signs so Streamlit markdown doesn't render them as LaTeX."""
    return text.replace("$", "\\$")

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
        st.markdown(_escape_dollars(msg["content"]))
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

    Protocol events:
      ": keepalive"                          → ignored comment (connection keepalive)
      data: {"token": "..."}                 → text token
      data: {"chart_part": "...", ...}       → chunked chart JSON fragment
      data: [DONE]{...}                      → terminal event
    """
    chart_parts: dict[int, str] = {}
    total_chart_parts: int = 0

    with httpx.stream(
        "POST",
        f"{API_BASE}/stream",
        json={"question": question, "session_id": session_id},
        timeout=httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            # SSE comment lines (keepalive) — skip silently
            if line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue

            payload = line[len("data:"):].strip()

            # ── Terminal event ────────────────────────────────────────────────
            if payload.startswith("[DONE]"):
                meta_json = payload[len("[DONE]"):]
                try:
                    meta = json.loads(meta_json)
                except Exception:
                    meta = {}

                # Reassemble chart_spec from collected parts
                chart_spec = None
                if chart_parts and total_chart_parts:
                    try:
                        full_json  = "".join(chart_parts[i] for i in range(total_chart_parts))
                        chart_spec = json.loads(full_json)
                    except Exception:
                        chart_spec = None

                meta["chart_spec"] = chart_spec
                yield meta
                return

            # ── Regular data event ────────────────────────────────────────────
            try:
                chunk = json.loads(payload)
            except Exception:
                continue

            # Chart chunk
            if "chart_part" in chunk:
                idx   = chunk.get("part_idx", 0)
                total = chunk.get("total_parts", 1)
                chart_parts[idx]   = chunk["chart_part"]
                total_chart_parts  = total
                continue

            # Text token
            yield chunk.get("token", "")


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

        full_answer          = ""
        chart_spec           = None
        first_token_received = False
        thinking_idx         = random.randint(0, len(THINKING_MESSAGES) - 1)
        last_cycle_time      = time.time()

        # Show initial thinking animation
        answer_placeholder.markdown(
            thinking_html(THINKING_MESSAGES[thinking_idx]),
            unsafe_allow_html=True,
        )

        try:
            for chunk in stream_from_api(user_input, st.session_state.session_id):
                if isinstance(chunk, str) and chunk:
                    if not first_token_received:
                        first_token_received = True
                    full_answer += chunk
                    answer_placeholder.markdown(_escape_dollars(full_answer) + "▌")
                elif isinstance(chunk, str) and not chunk:
                    # Empty token — cycle thinking message every ~2 seconds
                    if not first_token_received and (time.time() - last_cycle_time) > 2.0:
                        thinking_idx = (thinking_idx + 1) % len(THINKING_MESSAGES)
                        answer_placeholder.markdown(
                            thinking_html(THINKING_MESSAGES[thinking_idx]),
                            unsafe_allow_html=True,
                        )
                        last_cycle_time = time.time()
                elif isinstance(chunk, dict):
                    # Final DONE event
                    chart_spec = chunk.get("chart_spec")
                    if chunk.get("session_id"):
                        st.session_state.session_id = chunk["session_id"]

            # Remove cursor, show final answer
            answer_placeholder.markdown(_escape_dollars(full_answer))

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
