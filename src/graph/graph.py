"""
FinSight LangGraph Workflow
────────────────────────────
Topology:

  START
    │
    ▼
  analyze           ← rephrase + classify in one LLM call
    │
    ├─ csv_query   ──► csv_node    ──────────────────────► END
    ├─ sec_rag     ──► sec_node    ──────────────────────► END
    ├─ chart       ──► chart_node  ──────────────────────► END
    └─ hybrid      ──► csv_node ─┐
                     sec_node  ─┴──► synthesizer_node ──► END
"""

import logging
from langgraph.graph import StateGraph, START, END

from src.schemas import FinSightState
from src.nodes import (
    analyze_node,
    csv_node,
    sec_node,
    chart_node,
    synthesizer_node,
)

logger = logging.getLogger(__name__)


# ── Routing ───────────────────────────────────────────────────────────────────

def route_intent(state: FinSightState) -> str:
    intent = state.get("intent", "sec_rag")
    logger.debug("Routing → %s", intent)
    if intent == "csv_query":
        return "csv"
    if intent == "chart":
        return "chart"
    if intent == "hybrid":
        return "csv_hybrid"
    return "sec"  # sec_rag and any unrecognised value



# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(FinSightState)

    # Register nodes
    builder.add_node("analyze",     analyze_node)
    builder.add_node("csv",         csv_node)
    builder.add_node("sec",         sec_node)
    builder.add_node("chart",       chart_node)
    builder.add_node("csv_hybrid",  csv_node)
    builder.add_node("rag_hybrid",  sec_node)
    builder.add_node("synthesizer", synthesizer_node)

    # Entry
    builder.add_edge(START, "analyze")

    # Branch from analyze
    builder.add_conditional_edges(
        "analyze",
        route_intent,
        {
            "csv":        "csv",
            "sec":        "sec",
            "chart":      "chart",
            "csv_hybrid": "csv_hybrid",
        },
    )

    # Simple intent → END
    builder.add_edge("csv",   END)
    builder.add_edge("sec",   END)
    builder.add_edge("chart", END)

    # Hybrid path: csv_hybrid → rag_hybrid → synthesizer → END
    builder.add_edge("csv_hybrid",  "rag_hybrid")
    builder.add_edge("rag_hybrid",  "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile()


# Module-level compiled graph
graph = build_graph()
