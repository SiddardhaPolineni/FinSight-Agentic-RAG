"""
FinSight LangGraph Nodes
─────────────────────────
Each function receives and returns FinSightState dict slices.

Node execution order:
  rephrase_node → intent_node → csv_node       (csv_query)
                              → sec_node        (sec_rag)
                              → chart_node      (chart)
                              → csv_node +
                                sec_node +
                                synthesizer_node (hybrid)
"""

import json
import logging

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

import config as cfg
from src.prompts import (
    REPHRASE_PROMPT,
    INTENT_PROMPT,
    PANDAS_QUERY_PROMPT,
    CHART_QUERY_PROMPT,
    CSV_ANALYST_PROMPT,
    SEC_ANALYST_PROMPT,
    CHART_PROMPT,
    SYNTHESIZER_PROMPT,
)
from src.schemas import FinSightState
from src.utils import load_dataframes, sanitize_statement_type
from src.retriever import HybridRetriever, SchemaRetriever

logger = logging.getLogger(__name__)

# ── Singleton LLM, retrievers (created once per process) ─────────────────────
llm = ChatOpenAI(
    model=cfg.LLM_MODEL,
    temperature=cfg.LLM_TEMPERATURE,
    max_tokens=cfg.LLM_MAX_TOKENS,
    openai_api_key=cfg.OPENAI_API_KEY,
)

hybrid_retriever = HybridRetriever()
schema_retriever = SchemaRetriever()


def _history_text(messages: list) -> str:
    """Summarise the last 3 turns for the rephrase prompt."""
    lines = []
    for m in messages[-6:]:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content[:200]}")
    return "\n".join(lines) if lines else "None"


# ── Node 1: Rephrase ──────────────────────────────────────────────────────────

def rephrase_node(state: FinSightState) -> dict:
    """Rewrite the question into a self-contained query, resolving follow-up references."""
    history_text = _history_text(state.get("messages", []))
    try:
        resp = (REPHRASE_PROMPT | llm).invoke({
            "question": state["question"],
            "history":  history_text,
        })
        rephrased = resp.content.strip()
        logger.info("Rephrased: '%s' → '%s'", state["question"][:60], rephrased[:60])
        return {"rephrased_query": rephrased}
    except Exception as e:
        logger.error("Rephrase failed: %s", e)
        return {"rephrased_query": state["question"], "error": str(e)}


# ── Node 2: Intent classifier ─────────────────────────────────────────────────

def intent_node(state: FinSightState) -> dict:
    """Classify intent and extract entities."""
    question = state.get("rephrased_query") or state["question"]
    try:
        resp = (INTENT_PROMPT | llm).invoke({"question": question})
        raw  = resp.content.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)

        intent         = parsed.get("intent", "csv_query")
        companies      = [c.lower() for c in parsed.get("companies", [])]
        years          = [str(y) for y in parsed.get("years", [])]
        metrics        = parsed.get("metrics", [])
        statement_type = sanitize_statement_type(parsed.get("statement_type"))
        chart_type     = parsed.get("chart_type") or None
        if chart_type in ("null", "none", "", "None"):
            chart_type = None

        if intent not in {"csv_query", "sec_rag", "chart", "hybrid"}:
            intent = "csv_query"

        logger.info(
            "Intent: %s | companies: %s | years: %s | statement: %s",
            intent, companies, years, statement_type,
        )
        return {
            "intent":         intent,
            "companies":      companies,
            "years":          years,
            "metrics":        metrics,
            "statement_type": statement_type,
            "chart_type":     chart_type,
        }
    except Exception as e:
        logger.error("Intent classification failed: %s", e)
        return {
            "intent":         "csv_query",
            "companies":      [],
            "years":          [],
            "metrics":        [],
            "statement_type": None,
            "chart_type":     None,
            "error":          str(e),
        }


# ── Node 3: CSV query ─────────────────────────────────────────────────────────

def csv_node(state: FinSightState) -> dict:
    """
    Strictly data-first: every check gates the next step.
    Returns a clear "no data" message at the first missing piece
    rather than proceeding with incomplete information.
    """
    question       = state.get("rephrased_query") or state["question"]
    companies      = state.get("companies") or []
    years          = state.get("years") or None
    statement_type = sanitize_statement_type(state.get("statement_type"))

    # Gate 1 — must know which statement type
    if not statement_type:
        return {
            "csv_result": None,
            "answer": "I couldn't determine which financial statement to query. Could you clarify — are you asking about the income statement, balance sheet, or cash flow?",
        }

    # Gate 2 — must have at least one supported company
    supported = set(cfg.SUPPORTED_COMPANIES)
    valid_companies = [c for c in companies if c.lower() in supported]
    if not valid_companies:
        available = ", ".join(c.title() for c in cfg.SUPPORTED_COMPANIES)
        return {
            "csv_result": None,
            "answer": f"I only have financial data for {available}. Please ask about one of these companies.",
        }

    logger.info("CSV node — companies=%s, years=%s, statement=%s", valid_companies, years, statement_type)

    try:
        # Step 1 — load DataFrame
        df = load_dataframes(companies=valid_companies, statement_type=statement_type, years=years)

        # Gate 3 — DataFrame must have rows
        if df.empty:
            year_str = f" for {', '.join(years)}" if years else ""
            company_str = ", ".join(c.title() for c in valid_companies)
            return {
                "csv_result": None,
                "answer": f"I don't have {statement_type.replace('_', ' ')} data for {company_str}{year_str}.",
            }

        logger.info("DataFrame: %d rows | fiscalYears: %s", len(df), df["fiscalYear"].unique().tolist())

        # Step 2 — retrieve relevant column schema from Pinecone
        schema_ctx = schema_retriever.get_schema(
            question=question,
            statement_type=statement_type,
            top_k=10,
        )

        # Step 3 — LLM writes pandas expression
        resp        = (PANDAS_QUERY_PROMPT | llm).invoke({"column_context": schema_ctx, "question": question})
        pandas_expr = resp.content.strip().strip("```python").strip("```").strip()
        logger.info("Pandas expression: %s", pandas_expr)

        # Step 4 — execute
        result = eval(pandas_expr, {"df": df, "pd": pd})  # noqa: S307

        # Gate 4 — result must have actual data
        if result is None:
            return {
                "csv_result": None,
                "answer": "The query returned no results. The data may not be available for the requested company or time period.",
            }

        result_str = str(result).strip()

        if result_str in ("", "nan", "None", "NaN"):
            return {
                "csv_result": None,
                "answer": "The requested data is not available (reported as null in the source data).",
            }

        logger.info("Result: %s", result_str[:300])

        # Step 5 — format as natural language
        analyst_resp = (CSV_ANALYST_PROMPT | llm).invoke({"question": question, "data": result_str})
        return {
            "csv_result": result_str,
            "answer":     analyst_resp.content.strip(),
        }

    except ValueError as e:
        # load_dataframes raises ValueError when company/statement has no CSV
        logger.warning("CSV node — no data: %s", e)
        return {
            "csv_result": None,
            "answer": f"I don't have the data to answer this question. {e}",
        }
    except Exception as e:
        logger.error("CSV node failed: %s", e, exc_info=True)
        return {
            "csv_result": None,
            "answer": "Something went wrong while querying the financial data. Please try rephrasing your question.",
            "error": str(e),
        }


# ── Node 4: SEC RAG ───────────────────────────────────────────────────────────

def sec_node(state: FinSightState) -> dict:
    """Hybrid BM25 + Pinecone retrieval over SEC 10-K PDFs, with Cohere reranking."""
    query     = state.get("rephrased_query") or state["question"]
    companies = state.get("companies") or []
    years     = state.get("years") or []

    try:
        docs = hybrid_retriever.retrieve(
            query=query,
            company_filter=companies if companies else None,
            year_filter=years if years else None,
        )

        if not docs:
            return {
                "retrieved_docs": [],
                "sec_answer":     "No relevant SEC filing excerpts found for this query.",
            }

        context = "\n\n---\n\n".join(
            f"[Doc {i} | {d['company'].title()} | {d['year']} 10-K | score={d['score']:.4f}]\n{d['content']}"
            for i, d in enumerate(docs, 1)
        )

        resp   = (SEC_ANALYST_PROMPT | llm).invoke({"context": context, "question": state["question"]})
        answer = resp.content.strip()
        return {"retrieved_docs": docs, "sec_answer": answer, "answer": answer}

    except Exception as e:
        logger.error("SEC node failed: %s", e)
        return {"retrieved_docs": [], "sec_answer": f"Error: {e}", "error": str(e)}


# ── Node 5: Chart ─────────────────────────────────────────────────────────────

def chart_node(state: FinSightState) -> dict:
    """Build a Plotly chart from CSV data using LLM-specified axes."""
    import plotly.express as px

    question       = state.get("rephrased_query") or state["question"]
    companies      = state.get("companies") or cfg.SUPPORTED_COMPANIES
    years          = state.get("years") or None
    statement_type = sanitize_statement_type(state.get("statement_type"))

    if not statement_type:
        return {
            "chart_spec": None,
            "answer": "I couldn't determine which financial statement to use for this chart. Could you clarify — are you asking about the income statement, balance sheet, or cash flow?",
        }

    chart_type = state.get("chart_type") or "bar"

    try:
        # Step 1 — load DataFrame
        df = load_dataframes(companies=companies, statement_type=statement_type, years=years)

        # Step 2 — retrieve schema and ask LLM for chart spec
        schema_ctx = schema_retriever.get_schema(
            question=question,
            statement_type=statement_type,
            top_k=10,
        )
        resp = (CHART_QUERY_PROMPT | llm).invoke({"column_context": schema_ctx, "question": question})
        raw        = resp.content.strip().strip("```json").strip("```").strip()
        spec       = json.loads(raw)

        pandas_expr = spec["expr"]
        x_col       = spec["x"]
        y_col       = spec["y"]
        color_col   = spec.get("color") or None
        if color_col in ("null", "none", "None", ""):
            color_col = None
        title = spec.get("title", question[:60])

        logger.info("Chart spec — x=%s | y=%s | color=%s | expr=%s", x_col, y_col, color_col, pandas_expr)

        # Step 3 — execute expression
        result_df = eval(pandas_expr, {"df": df, "pd": pd})  # noqa: S307

        if not isinstance(result_df, pd.DataFrame):
            result_df = result_df.reset_index()
            result_df.columns = [x_col, y_col]

        result_df = result_df.reset_index(drop=True)

        # Validate columns exist
        for col in [c for c in [x_col, y_col, color_col] if c]:
            if col not in result_df.columns:
                raise ValueError(f"Column '{col}' not in result. Available: {list(result_df.columns)}")

        # Step 4 — build Plotly figure
        axis_labels = {
            x_col: x_col.replace("_", " ").title(),
            y_col: y_col.replace("_", " ").title(),
        }
        if chart_type == "line":
            fig = px.line(result_df, x=x_col, y=y_col, color=color_col, title=title, markers=True, labels=axis_labels)
        else:
            fig = px.bar(result_df, x=x_col, y=y_col, color=color_col, barmode="group", title=title, labels=axis_labels)

        fig.update_layout(
            template="plotly_white",
            legend_title=color_col.replace("_", " ").title() if color_col else "",
            xaxis={"type": "category"},
            yaxis_tickformat="$.3s",
        )

        # Step 5 — describe chart insights
        resp   = (CHART_PROMPT | llm).invoke({
            "data_summary": result_df.to_markdown(index=False)[:800],
            "question":     question,
        })
        return {
            "chart_spec": fig.to_dict(),
            "answer":     resp.content.strip(),
        }

    except Exception as e:
        logger.error("Chart node failed: %s", e, exc_info=True)
        return {
            "chart_spec": None,
            "answer":     f"Chart generation failed: {e}",
            "error":      str(e),
        }


# ── Node 6: Synthesizer ───────────────────────────────────────────────────────

def synthesizer_node(state: FinSightState) -> dict:
    """Combine CSV and SEC outputs into a single polished response (hybrid intent only)."""
    csv_result = state.get("csv_result") or "N/A"
    sec_result = state.get("sec_answer") or "N/A"

    # If only one source has data, pass it through directly
    if csv_result == "N/A":
        return {"answer": sec_result}
    if sec_result == "N/A":
        return {"answer": state.get("answer") or csv_result}

    try:
        resp = (SYNTHESIZER_PROMPT | llm).invoke({
            "question":   state["question"],
            "csv_result": csv_result,
            "rag_result": sec_result,
        })
        return {"answer": resp.content.strip()}
    except Exception as e:
        logger.error("Synthesizer failed: %s", e)
        return {"answer": state.get("sec_answer") or state.get("csv_result") or "Error generating response."}
