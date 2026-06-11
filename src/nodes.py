"""
FinSight LangGraph Nodes
─────────────────────────
Each function receives and returns FinSightState dict slices.

Node execution order:
  rephrase_node → intent_node → retriever_node (conditional)
                               → csv_node       (conditional)
                               → chart_node     (conditional)
                               → synthesizer_node → END
"""

import json
import logging
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src import config as cfg
from src.prompts import (
    REPHRASE_PROMPT, INTENT_PROMPT,
    PANDAS_QUERY_PROMPT, CHART_QUERY_PROMPT, CSV_ANALYST_PROMPT,
    RAG_ANALYST_PROMPT, CHART_PROMPT, SYNTHESIZER_PROMPT,
)
from src.state import FinSightState
from src.csv_engine import load_dataframes
from src.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# ── Singleton LLM and retriever (created once per process) ────────────────────
llm = ChatOpenAI(
    model=cfg.LLM_MODEL,
    temperature=cfg.LLM_TEMPERATURE,
    max_tokens=cfg.LLM_MAX_TOKENS,
    openai_api_key=cfg.OPENAI_API_KEY,
)

retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global retriever
    if retriever is None:
        retriever = HybridRetriever()
    return retriever


def _history_text(messages: list) -> str:
    """Convert message list to a short readable string for prompts."""
    lines = []
    for m in messages[-6:]:   # last 3 turns
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content[:200]}")
    return "\n".join(lines) if lines else "None"


# ── Node 1: Rephrase ──────────────────────────────────────────────────────────

def rephrase_node(state: FinSightState) -> dict:
    """Rewrite the current question for better retrieval"""
    chain = REPHRASE_PROMPT | llm
    try:
        resp = chain.invoke({"question": state["question"]})
        rephrased = resp.content.strip()
        logger.info("Rephrased: '%s' → '%s'", state["question"][:60], rephrased[:60])
        return {"rephrased_query": rephrased}
    except Exception as e:
        logger.error("Rephrase failed: %s", e)
        return {"rephrased_query": state["question"], "error": str(e)}


# ── Node 2: Intent classifier ─────────────────────────────────────────────────

def intent_node(state: FinSightState) -> dict:
    """Classify intent and extract entities (companies, years, metrics, statement type)."""
    question = state.get("rephrased_query") or state["question"]
    chain = INTENT_PROMPT | llm
    try:
        resp = chain.invoke({"question": question})
        print("intent response", resp)
        raw = resp.content.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)

        intent         = parsed.get("intent", "csv_query")
        companies      = [c.lower() for c in parsed.get("companies", [])]
        years          = [str(y) for y in parsed.get("years", [])]
        metrics        = parsed.get("metrics", [])
        statement_type = parsed.get("statement_type") or None
        chart_type     = parsed.get("chart_type") or None

        # Validate intent
        valid_intents = {"csv_query", "sec_rag", "chart", "hybrid"}
        if intent not in valid_intents:
            intent = "csv_query"


        return {
            "intent": intent,
            "companies": companies,
            "years": years,
            "metrics": metrics,
            "statement_type": statement_type,
            "chart_type": chart_type,
        }
    except Exception as e:
        logger.error("Intent classification failed: %s", e)
        return {
            "intent": "csv_query",
            "companies": [],
            "years": [],
            "metrics": [],
            "statement_type": None,
            "chart_type": None,
            "error": str(e),
        }


# ── Node 3: CSV query ─────────────────────────────────────────────────────────

def csv_node(state: FinSightState) -> dict:
    """
    LLM writes a pandas expression that computes the answer directly.
    The result (scalar, Series, or small DataFrame) is stringified and
    passed to the LLM only for natural-language wrapping — not for analysis.
    """
    question       = state.get("rephrased_query") or state["question"]
    companies      = state.get("companies") or cfg.SUPPORTED_COMPANIES
    years          = state.get("years") or None
    statement_type = state.get("statement_type") or "income_statement"

    try:
        # Step 1 — load DataFrame + schema description
        df, schema_desc = load_dataframes(
            companies=companies,
            statement_type=statement_type,
            years=years,
        )

        # Step 2 — LLM generates a pandas expression that computes the answer
        query_chain = PANDAS_QUERY_PROMPT | llm
        query_resp  = query_chain.invoke({"schema": schema_desc, "question": question})
        pandas_expr = query_resp.content.strip().strip("```python").strip("```").strip()
        logger.info("Pandas expression: %s", pandas_expr)

        # Step 3 — execute and get the computed result
        result = eval(pandas_expr, {"df": df, "pd": pd})
        result_str = str(result)
        logger.info("Query result: %s", result_str[:200])

        # Step 4 — LLM wraps the raw result in a clean sentence
        analyst_chain = CSV_ANALYST_PROMPT | llm
        analyst_resp  = analyst_chain.invoke({"question": question, "data": result_str})
        answer = analyst_resp.content.strip()

        return {
            "csv_result": result_str,
            "answer":     answer,
        }

    except Exception as e:
        logger.error("CSV node failed: %s", e, exc_info=True)
        return {
            "csv_result": f"Error: {e}",
            "answer":     f"I encountered an error querying the financial data: {e}",
            "error":      str(e),
        }


# ── Node 4: RAG query ─────────────────────────────────────────────────────────

def sec_node(state: FinSightState) -> dict:
    """Hybrid retrieval (BM25 + Pinecone + Cohere reranker) from SEC 10-K PDFs."""
    query     = state.get("rephrased_query") or state["question"]
    companies = state.get("companies", [])
    years     = state.get("years", [])

    try:
        retriever = _get_retriever()
        docs = retriever.retrieve(
            query=query,
            company_filter=companies if companies else None,
            year_filter=years if years else None,
        )

        if not docs:
            return {
                "retrieved_docs": [],
                "rag_answer": "No relevant SEC filing excerpts found for this query.",
            }

        # Build context for LLM
        context_parts = []
        for i, d in enumerate(docs, 1):
            header = f"[Doc {i} | {d['company'].title()} | {d['year']} 10-K | score={d['score']:.4f}]"
            context_parts.append(f"{header}\n{d['content']}")
        context = "\n\n---\n\n".join(context_parts)

        chain = SEC_ANALYST_PROMPT | llm
        resp  = chain.invoke({"context": context, "question": state["question"]})
        answer = resp.content.strip()

        return {"retrieved_docs": docs, "sec_answer": answer, "answer": answer}

    except Exception as e:
        logger.error("SEC node failed: %s", e)
        return {"retrieved_docs": [], "sec_answer": f"Error: {e}", "error": str(e)}


# ── Node 5: Chart node ────────────────────────────────────────────────────────

def chart_node(state: FinSightState) -> dict:
    """
    Builds a Plotly chart from CSV data.

    The LLM is asked to produce:
      - a pandas expression that returns a plot-ready DataFrame
      - which column is x, which is y, which is color (all decided by the LLM
        based on what the question asks — not hardcoded here)

    This handles all axis combinations:
      - fiscalYear on x  → trend over time
      - company on x     → cross-company comparison / cumulative
      - fiscalYear as color when comparing companies over time
    """
    import plotly.express as px

    question       = state.get("rephrased_query") or state["question"]
    companies      = state.get("companies") or cfg.SUPPORTED_COMPANIES
    years          = state.get("years") or None
    statement_type = state.get("statement_type") or "income_statement"
    chart_type     = state.get("chart_type") or "bar"

    try:
        # Step 1 — load DataFrame + schema
        df, schema_desc = load_dataframes(
            companies=companies,
            statement_type=statement_type,
            years=years,
        )

        # Step 2 — LLM returns pandas expr + axis spec as JSON
        query_chain = CHART_QUERY_PROMPT | llm
        query_resp  = query_chain.invoke({"schema": schema_desc, "question": question})
        raw = query_resp.content.strip().strip("```json").strip("```").strip()

        spec = json.loads(raw)
        pandas_expr = spec["expr"]
        x_col       = spec["x"]
        y_col       = spec["y"]
        color_col   = spec.get("color") or None
        title       = spec.get("title", question[:60])

        logger.info("Chart spec — expr: %s | x=%s | y=%s | color=%s",
                    pandas_expr, x_col, y_col, color_col)

        # Step 3 — execute the expression
        result_df = eval(pandas_expr, {"df": df, "pd": pd})  # noqa: S307

        if not isinstance(result_df, pd.DataFrame):
            # LLM returned a Series — convert to DataFrame
            result_df = result_df.reset_index()
            result_df.columns = [x_col, y_col]

        result_df = result_df.reset_index(drop=True)

        # Verify the expected columns actually exist
        for col in [c for c in [x_col, y_col, color_col] if c]:
            if col not in result_df.columns:
                raise ValueError(
                    f"Column '{col}' not found in result DataFrame. "
                    f"Available: {list(result_df.columns)}"
                )

        # Step 4 — build Plotly figure using LLM-specified axes
        axis_labels = {x_col: x_col.replace("_", " ").title(),
                       y_col: y_col.replace("_", " ").title()}

        if chart_type == "line":
            fig = px.line(
                result_df, x=x_col, y=y_col, color=color_col,
                title=title, markers=True, labels=axis_labels,
            )
        else:  # bar (default)
            fig = px.bar(
                result_df, x=x_col, y=y_col, color=color_col,
                barmode="group", title=title, labels=axis_labels,
            )

        fig.update_layout(
            template="plotly_white",
            legend_title=color_col.replace("_", " ").title() if color_col else "",
            xaxis={"type": "category"},
            yaxis_tickformat="$.3s",
        )

        chart_spec = fig.to_dict()

        # Step 5 — LLM describes chart insights
        chain  = CHART_PROMPT | llm
        resp   = chain.invoke({
            "data_summary": result_df.to_markdown(index=False)[:800],
            "question":     question,
        })
        answer = resp.content.strip()

        return {
            "chart_spec": chart_spec,
            "answer":     answer,
        }

    except Exception as e:
        logger.error("Chart node failed: %s", e, exc_info=True)
        return {
            "chart_spec": None,
            "answer":     f"Chart generation failed: {e}",
            "error":      str(e),
        }


# ── Node 6: Synthesizer ────────────────────────────────────────────────────────

def synthesizer_node(state: FinSightState) -> dict:
    """
    Combine CSV and SEC outputs into a single polished response.
    Only invoked for 'hybrid' intent where both sources contribute.
    """
    csv_result = state.get("csv_result") or "N/A"
    sec_result = state.get("sec_answer") or "N/A"

    # If one source is N/A, just forward the available answer
    if csv_result == "N/A" and sec_result != "N/A":
        return {"answer": rag_result}
    if sec_result == "N/A" and csv_result != "N/A":
        answer = state.get("answer") or csv_result
        return {"answer": answer}

    chain = SYNTHESIZER_PROMPT | llm
    try:
        resp = chain.invoke({
            "question":   state["question"],
            "csv_result": csv_result,
            "rag_result": rag_result,
        })
        return {"answer": resp.content.strip()}
    except Exception as e:
        logger.error("Synthesizer failed: %s", e)
        return {"answer": state.get("rag_answer") or state.get("csv_result") or "Error generating response."}
