"""
LangGraph state — single TypedDict that flows through every node.
"""

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


# Intent types the classifier can return
IntentType = str   # "csv_query" | "sec_rag" | "chart" | "hybrid"


class FinSightState(TypedDict):
    # ── Conversation ──────────────────────────────────────────────────────────
    messages: Annotated[list, add_messages]   # full chat history (LangGraph managed)
    question: str                              # current raw user question

    # ── Rephrase & intent ─────────────────────────────────────────────────────
    rephrased_query:    Optional[str]
    intent:             Optional[IntentType]   # csv_query | sec_rag | chart | hybrid
    companies:          list[str]              # ["apple", "nvidia", ...]
    years:              list[str]              # ["2023", "2024", ...]
    metrics:            list[str]              # ["revenue", "netIncome", ...]
    statement_type:     Optional[str]          # income_statement | balance_sheet | cash_flow
    chart_type:         Optional[str]          # bar | line | pie (for chart intent)

    # ── CSV query results ─────────────────────────────────────────────────────
    csv_result:         Optional[str]          # markdown table / summary
    csv_dataframe_json: Optional[str]          # JSON for chart rendering

    # ── RAG results ───────────────────────────────────────────────────────────
    retrieved_docs:     list[dict]             # {content, source, company, year, score}
    sec_answer:         Optional[str]

    # ── Chart ─────────────────────────────────────────────────────────────────
    chart_spec:         Optional[dict]         # plotly figure dict (json-serialisable)

    # ── Final answer ──────────────────────────────────────────────────────────
    answer:             Optional[str]

    # ── Control ───────────────────────────────────────────────────────────────
    error:              Optional[str]
