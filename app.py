"""
FinSight — Streamlit UI
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
    page_title="FinSight",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000/FinSight"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar colored ribbon header */
[data-testid="stSidebar"] {
    background-color: #f0fdf4;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
.ribbon {
    background: linear-gradient(135deg, #059669, #10b981);
    padding: 0.8rem 1rem;
    margin: -1rem -1rem 1rem -1rem;
    border-radius: 0 0 12px 12px;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.ribbon .logo { font-size: 1.6rem; }
.ribbon .title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
}
.ribbon .subtitle {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.8);
    margin: 0;
}

/* Thinking dots */
@keyframes dot-bounce {
    0%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-5px); }
}
.thinking-box {
    display: flex; align-items: center; gap: 10px; padding: 6px 0; opacity: 0.7;
}
.thinking-box .dots { display: flex; gap: 4px; }
.thinking-box .dots span {
    width: 6px; height: 6px; border-radius: 50%;
    background: #6366f1; display: inline-block;
    animation: dot-bounce 1.2s ease-in-out infinite;
}
.thinking-box .dots span:nth-child(2) { animation-delay: 0.15s; }
.thinking-box .dots span:nth-child(3) { animation-delay: 0.3s; }
.thinking-box .msg { color: #6b7280; font-style: italic; font-size: 0.9rem; }

/* Chat input — clean single box, no inner highlight */
[data-testid="stChatInput"] {
    background-color: #f9fafb;
    border: 1px solid #d1d5db;
    border-radius: 8px;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:active,
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div:focus-within {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    background: transparent !important;
}
/* Color the send/submit button */
[data-testid="stChatInput"] button {
    background-color: #059669 !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    border: none !important;
}
[data-testid="stChatInput"] button:hover {
    background-color: #047857 !important;
}

/* Push main content to top */
.block-container {
    padding-top: 1rem !important;
}

/* Hide streamlit branding */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

THINKING_MESSAGES = [
    "Searching financial records",
    "Analyzing statements",
    "Crunching the numbers",
    "Scanning SEC filings",
    "Synthesizing insights",
    "Preparing your answer",
]


def _thinking(msg: str) -> str:
    return (
        '<div class="thinking-box">'
        '<div class="dots"><span></span><span></span><span></span></div>'
        f'<span class="msg">{msg}…</span>'
        '</div>'
    )


def _esc(text: str) -> str:
    return text.replace("$", "\\$")


# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Colored ribbon header
    st.markdown("""
    <div class="ribbon">
        <div class="logo">📈</div>
        <div>
            <div class="title">FinSight</div>
            <div class="subtitle">AI-Powered Financial Research</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:0.8rem;">
        <p style="font-size:0.75rem; font-weight:600; color:#6b7280; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:0.5rem;">📊 Data Coverage</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.85rem; line-height:1.7; color:#374151;">
        <b>Financial Statements</b><br/>
        <span style="color:#6b7280;">Companies:</span> Apple · Google · Microsoft · Nvidia<br/>
        <span style="color:#6b7280;">Statements:</span> Income · Balance Sheet · Cash Flow<br/>
        <span style="color:#6b7280;">Period:</span> FY 2021 – 2025<br/><br/>
        <b>SEC 10-K Filings</b><br/>
        <span style="color:#6b7280;">Companies:</span> Apple · Google · Microsoft · NVIDIA<br/>
        <span style="color:#6b7280;">Years:</span> 2023 · 2024 · 2025
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <p style="font-size:0.75rem; font-weight:600; color:#6b7280; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:0.5rem;">💡 Try Asking</p>
    """, unsafe_allow_html=True)

    examples = [
        "What was Nvidia's revenue in 2024?",
        "Compare Apple and Microsoft net income for 2024",
        "What are Google's key risk factors in its 2024 10-K?",
        "Plot Nvidia's free cash flow from 2021 to 2025",
        "Show a bar chart of revenue for all companies in 2024",
        "What did Microsoft say about AI strategy in 2024?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:30]}"):
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


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem 0;">
    <span style="font-size:2.5rem;">📈</span>
    <h1 style="font-size:1.8rem; font-weight:700; color:#1f2937; margin:0.3rem 0 0.2rem 0;">FinSight</h1>
    <p style="font-size:0.95rem; color:#6b7280; margin:0;">Your AI-powered financial research assistant</p>
    <p style="font-size:0.85rem; color:#9ca3af; margin:0.8rem 0 0 0;">
        Ask about revenue, margins, growth trends, SEC 10-K filings, or request charts.
    </p>
</div>
""", unsafe_allow_html=True)

# Sample question chips (only show when no chat history)
if not st.session_state.chat_history:
    st.markdown("""
    <div style="text-align:center; margin-bottom:1.5rem;">
        <p style="font-size:0.78rem; color:#9ca3af; margin-bottom:0.6rem;">Try one of these:</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    quick_questions = [
        ("What was Nvidia's revenue in 2024?", col1),
        ("Compare Apple and Microsoft net income for 2024", col2),
        ("Plot a bar chart of revenue for all companies", col1),
        ("What did Microsoft say about AI in its 10-K?", col2),
    ]
    for q, col in quick_questions:
        with col:
            if st.button(q, use_container_width=True, key=f"main_{q[:25]}"):
                st.session_state["prefill"] = q
                st.rerun()

    st.divider()
else:
    st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(_esc(msg["content"]))
        if msg.get("chart_spec"):
            st.plotly_chart(
                go.Figure(msg["chart_spec"]),
                use_container_width=True,
                key=f"hist_{id(msg)}",
            )


# ── Stream helper ─────────────────────────────────────────────────────────────

def stream_from_api(question: str, session_id: str):
    chart_parts: dict[int, str] = {}
    total_parts: int = 0

    with httpx.stream(
        "POST", f"{API_BASE}/stream",
        json={"question": question, "session_id": session_id},
        timeout=httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0),
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line.startswith(":") or not line.startswith("data:"):
                continue
            payload = line[5:].strip()

            if payload.startswith("[DONE]"):
                meta = {}
                try:
                    meta = json.loads(payload[6:])
                except Exception:
                    pass
                chart_spec = None
                if chart_parts and total_parts:
                    try:
                        chart_spec = json.loads("".join(chart_parts[i] for i in range(total_parts)))
                    except Exception:
                        pass
                meta["chart_spec"] = chart_spec
                yield meta
                return

            try:
                chunk = json.loads(payload)
            except Exception:
                continue

            if "chart_part" in chunk:
                chart_parts[chunk.get("part_idx", 0)] = chunk["chart_part"]
                total_parts = chunk.get("total_parts", 1)
            else:
                yield chunk.get("token", "")


# ── Chat input ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Ask a financial question…")
if prefill and not user_input:
    user_input = prefill

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        answer_box = st.empty()
        chart_box = st.empty()

        full_answer = ""
        chart_spec = None
        got_token = False
        think_idx = random.randint(0, len(THINKING_MESSAGES) - 1)
        last_think = time.time()

        answer_box.markdown(_thinking(THINKING_MESSAGES[think_idx]), unsafe_allow_html=True)

        try:
            for chunk in stream_from_api(user_input, st.session_state.session_id):
                if isinstance(chunk, str) and chunk:
                    got_token = True
                    full_answer += chunk
                    answer_box.markdown(_esc(full_answer) + "▌")
                elif isinstance(chunk, str):
                    if not got_token and (time.time() - last_think) > 2.0:
                        think_idx = (think_idx + 1) % len(THINKING_MESSAGES)
                        answer_box.markdown(_thinking(THINKING_MESSAGES[think_idx]), unsafe_allow_html=True)
                        last_think = time.time()
                elif isinstance(chunk, dict):
                    chart_spec = chunk.get("chart_spec")
                    if chunk.get("session_id"):
                        st.session_state.session_id = chunk["session_id"]

            answer_box.markdown(_esc(full_answer))
            if chart_spec:
                with chart_box:
                    st.plotly_chart(go.Figure(chart_spec), use_container_width=True, key=f"c_{uuid.uuid4()}")

        except httpx.ConnectError:
            full_answer = "⚠️ Cannot connect to FinSight API. Start the backend with: `uvicorn api:app --reload --port 8000`"
            answer_box.error(full_answer)
        except Exception as e:
            full_answer = f"⚠️ Error: {e}"
            answer_box.error(full_answer)

        st.session_state.chat_history.append({"role": "assistant", "content": full_answer, "chart_spec": chart_spec})
    st.rerun()
