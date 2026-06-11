"""
FinSight LangGraph Workflow
────────────────────────────
Topology:

  START
    │
    ▼
  rephrase          ← clean & expand user question
    │
    ▼
  intent            ← classify: csv_query | sec_rag | chart | hybrid
    │
    ├─ csv_query   ──► csv_node    ──────────────────────► END
    ├─ sec_rag     ──► rag_node    ──────────────────────► END
    ├─ chart       ──► chart_node  ──────────────────────► END
    └─ hybrid      ──► csv_node ─┐
                     rag_node  ─┴──► synthesizer_node ──► END
"""

import logging
from langgraph.graph import StateGraph, START, END

from src.state import FinSightState
from src.nodes import (
    rephrase_node,
    intent_node,
    csv_node,
    sec_node,
    chart_node,
    synthesizer_node,
)

logger = logging.getLogger(__name__)


# ── Routing ───────────────────────────────────────────────────────────────────

def route_intent(state: FinSightState) -> str:
    intent = state.get("intent", "csv_query")
    logger.debug("Routing → %s", intent)
    if intent == "sec_rag":
        return "sec"
    if intent == "chart":
        return "chart"
    if intent == "hybrid":
        return "csv_hybrid"
    return "csv"


def route_after_csv_hybrid(state: FinSightState) -> str:
    """After csv_node in hybrid mode, always go to rag_node."""
    return "rag_hybrid"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(FinSightState)

    # Register nodes
    builder.add_node("rephrase",    rephrase_node)
    builder.add_node("intent",      intent_node)
    builder.add_node("csv",         csv_node)
    builder.add_node("sec",         sec_node)
    builder.add_node("chart",       chart_node)
    builder.add_node("csv_hybrid",  csv_node)
    builder.add_node("rag_hybrid",  rag_node)
    builder.add_node("synthesizer", synthesizer_node)

    # Linear entry
    builder.add_edge(START, "rephrase")
    builder.add_edge("rephrase", "intent")

    # Branch from intent
    builder.add_conditional_edges(
        "intent",
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
