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

import asyncio
import json
import logging

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

import config as cfg
from src.prompts import (
    ANALYZE_PROMPT,
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

# ── Singleton LLM instances (created once per process) ───────────────────────
# Full-power LLM for analysis and generation tasks
llm = ChatOpenAI(
    model=cfg.LLM_MODEL,
    temperature=cfg.LLM_TEMPERATURE,
    max_tokens=cfg.LLM_MAX_TOKENS,
    openai_api_key=cfg.OPENAI_API_KEY,
)

# Lightweight LLM for short formatting tasks (analyst narrator, chart describer)
# Capped at 512 tokens — these calls only produce 1–4 sentence summaries
llm_fast = ChatOpenAI(
    model=cfg.LLM_MODEL,
    temperature=0.0,
    max_tokens=512,
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


# ── Node 1: Analyze (rephrase + intent in one LLM call) ──────────────────────

def analyze_node(state: FinSightState) -> dict:
    """
    Single LLM call that rewrites the question for retrieval AND
    classifies intent + extracts entities simultaneously.
    Replaces the previous two-node rephrase → intent chain.
    """
    history_text = _history_text(state.get("messages", []))
    try:
        resp = (ANALYZE_PROMPT | llm).invoke({
            "question": state["question"],
            "history":  history_text,
        })
        raw    = resp.content.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)

        rephrased      = parsed.get("rephrased_query") or state["question"]
        intent         = parsed.get("intent", "sec_rag")
        companies      = [c.lower() for c in parsed.get("companies", [])]
        years          = [str(y) for y in parsed.get("years", [])]
        metrics        = parsed.get("metrics", [])
        statement_type = sanitize_statement_type(parsed.get("statement_type"))
        chart_type     = parsed.get("chart_type") or None
        if chart_type in ("null", "none", "", "None"):
            chart_type = None

        if intent not in {"csv_query", "sec_rag", "chart", "hybrid"}:
            intent = "sec_rag"

        logger.info(
            "Analyze: rephrased='%s' | intent=%s | companies=%s | years=%s | statement=%s",
            rephrased[:60], intent, companies, years, statement_type,
        )
        return {
            "rephrased_query": rephrased,
            "intent":          intent,
            "companies":       companies,
            "years":           years,
            "metrics":         metrics,
            "statement_type":  statement_type,
            "chart_type":      chart_type,
        }
    except Exception as e:
        logger.error("Analyze node failed: %s", e)
        return {
            "rephrased_query": state["question"],
            "intent":          "sec_rag",
            "companies":       [],
            "years":           [],
            "metrics":         [],
            "statement_type":  None,
            "chart_type":      None,
            "error":           str(e),
        }


# ── Node 3: CSV query ─────────────────────────────────────────────────────────

def csv_node(state: FinSightState) -> dict:
    """
    Strictly data-first: every check gates the next step.
    Returns a clear "no data" message at the first missing piece
    rather than proceeding with incomplete information.

    Optimisations:
    - schema fetch (Pinecone+embed) and DataFrame load run concurrently
    - pandas-expr LLM call fires as soon as schema is ready
    - CSV_ANALYST_PROMPT uses llm_fast (512 max_tokens) — just a formatter
    """
    question       = state.get("rephrased_query") or state["question"]
    companies      = state.get("companies") or []
    years          = state.get("years") or None
    statement_type = sanitize_statement_type(state.get("statement_type"))

    # Gate 1 — must know which statement type
    if not statement_type:
        # Try to infer from question/metrics before failing
        metrics_str = " ".join(state.get("metrics") or []).lower()
        q_lower     = (state.get("rephrased_query") or state["question"]).lower()
        combined    = metrics_str + " " + q_lower
        if any(kw in combined for kw in [
            "revenue", "net income", "gross profit", "operating income", "ebitda",
            "eps", "earnings per share", "cost of revenue", "r&d", "operating margin",
            "net margin", "profit margin",
        ]):
            statement_type = "income_statement"
        elif any(kw in combined for kw in [
            "free cash flow", "fcf", "operating cash", "capex", "capital expenditure",
            "cash from operations", "investing", "financing",
        ]):
            statement_type = "cash_flow"
        elif any(kw in combined for kw in [
            "total assets", "liabilities", "equity", "debt", "cash and cash equivalents",
            "goodwill", "working capital", "debt-to-equity",
        ]):
            statement_type = "balance_sheet"

    if not statement_type:
        return {
            "csv_result": None,
            "answer": "I couldn't determine which financial statement to query. Could you clarify — are you asking about the income statement, balance sheet, or cash flow?",
        }

    # Gate 2 — must have at least one supported company
    supported = set(cfg.SUPPORTED_COMPANIES)
    valid_companies = [c for c in companies if c.lower() in supported]
    if not valid_companies:
        # If the question implies a comparison across all companies, use all
        q_lower = question.lower()
        is_all_companies_query = any(kw in q_lower for kw in [
            "which company", "all companies", "each company", "every company",
            "among", "highest", "lowest", "most", "least", "best", "worst",
            "compare all", "rank",
        ])
        if is_all_companies_query:
            valid_companies = cfg.SUPPORTED_COMPANIES
        else:
            available = ", ".join(c.title() for c in cfg.SUPPORTED_COMPANIES)
            return {
                "csv_result": None,
                "answer": f"I only have financial data for {available}. Please ask about one of these companies.",
            }

    logger.info("CSV node — companies=%s, years=%s, statement=%s", valid_companies, years, statement_type)

    # Gate 3 — for specific metric queries, require a year
    # If user asks about a trend/comparison ("over the years", "from X to Y"), empty years is fine.
    # But for single-point questions ("What was X's revenue?"), ask which year.
    if not years:
        q_lower = question.lower()
        is_trend_query = any(kw in q_lower for kw in [
            "trend", "over the years", "growth", "from", "to", "all years",
            "compare", "comparison", "vs", "versus", "each year", "year over year",
            "yoy", "historical", "across", "consistent", "most", "highest",
            "lowest", "best", "worst", "strongest", "weakest",
        ])
        if not is_trend_query:
            company_str = ", ".join(c.title() for c in valid_companies)
            return {
                "csv_result": None,
                "answer": f"Which fiscal year are you asking about for {company_str}? I have data for 2021, 2022, 2023, 2024, and 2025.",
            }

    try:
        # Step 1 — load DataFrame and fetch schema concurrently
        # DataFrame loading is CPU/IO bound; schema fetch is network bound.
        # Run them in parallel with asyncio so neither blocks the other.
        async def _fetch_schema():
            return schema_retriever.get_schema(
                question=question,
                statement_type=statement_type,
                top_k=10,
            )

        async def _parallel():
            return await asyncio.gather(
                asyncio.to_thread(load_dataframes, valid_companies, statement_type, years),
                _fetch_schema(),
            )

        try:
            loop = asyncio.get_running_loop()
            # We're inside an async context (FastAPI) — use a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                df_future     = pool.submit(load_dataframes, valid_companies, statement_type, years)
                schema_future = pool.submit(schema_retriever.get_schema, question, statement_type, 10)
                df         = df_future.result()
                schema_ctx = schema_future.result()
        except RuntimeError:
            # No running loop — plain synchronous path
            df         = load_dataframes(valid_companies, statement_type, years)
            schema_ctx = schema_retriever.get_schema(question, statement_type, 10)

        # Gate 3 — DataFrame must have rows
        if df.empty:
            year_str    = f" for {', '.join(years)}" if years else ""
            company_str = ", ".join(c.title() for c in valid_companies)
            return {
                "csv_result": None,
                "answer": f"I don't have {statement_type.replace('_', ' ')} data for {company_str}{year_str}.",
            }

        logger.info("DataFrame: %d rows | fiscalYears: %s", len(df), df["fiscalYear"].unique().tolist())

        # Step 2 — LLM writes pandas expression + answer template in one call
        resp = (PANDAS_QUERY_PROMPT | llm).invoke({"column_context": schema_ctx, "question": question})
        raw_resp = resp.content.strip().strip("```json").strip("```").strip()
        try:
            llm_output = json.loads(raw_resp)
            pandas_expr     = llm_output["expr"]
            answer_template = llm_output.get("answer_template", "{result}")
        except (json.JSONDecodeError, KeyError):
            # Fallback — treat entire response as a pandas expression
            pandas_expr     = raw_resp.strip("```python").strip("```").strip()
            answer_template = "{result}"

        logger.info("Pandas expression: %s", pandas_expr)

        # Step 3 — execute
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

        # Step 4 — format answer
        # For simple scalar results: use the template directly (0 ms)
        # For complex results (DataFrames, rankings): use llm_fast with actual data (~1s)
        def _format_value(val_str: str) -> str:
            """Apply basic number formatting to make values readable."""
            try:
                f = float(val_str)
                abs_f = abs(f)
                q = question.lower()
                # Percentages
                if any(kw in q for kw in ["margin", "ratio", "rate", "growth", "return", "%"]):
                    return f"{f * 100:.2f}%" if abs(f) <= 10 else f"{f:.2f}%"
                # Large monetary values
                if abs_f >= 1e12:
                    return f"${f/1e12:.2f}T"
                if abs_f >= 1e9:
                    return f"${f/1e9:.2f}B"
                if abs_f >= 1e6:
                    return f"${f/1e6:.2f}M"
                return val_str
            except (ValueError, TypeError):
                return val_str

        is_complex_result = '\n' in result_str or len(result_str) > 100
        template_is_vague = any(phrase in answer_template.lower() for phrase in [
            "listed in the result", "shown below", "as follows", "in the result",
            "see the result", "the result shows",
        ])

        if is_complex_result or template_is_vague:
            # Complex result — LLM needs to interpret the data
            analyst_resp = (CSV_ANALYST_PROMPT | llm_fast).invoke({"question": question, "data": result_str})
            answer = analyst_resp.content.strip()
        else:
            # Simple scalar — template substitution is enough
            formatted_result = _format_value(result_str)
            # Avoid double dollar signs: if template already has $ before {result}
            if "${result}" in answer_template or "$ {result}" in answer_template:
                # Template has a dollar sign — don't add another in formatted_result
                formatted_result = formatted_result.lstrip("$")
            answer = answer_template.replace("{result}", formatted_result)

        return {
            "csv_result": result_str,
            "answer":     answer,
        }

    except ValueError as e:
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

    # ── Infer statement_type from metrics when the LLM didn't set it ──────────
    if not statement_type:
        metrics_str = " ".join(state.get("metrics") or []).lower()
        q_lower     = question.lower()
        combined    = metrics_str + " " + q_lower
        if any(kw in combined for kw in [
            "revenue", "net income", "gross profit", "operating income", "ebitda",
            "eps", "earnings per share", "cost of revenue", "r&d", "operating margin",
            "net margin", "profit margin",
        ]):
            statement_type = "income_statement"
        elif any(kw in combined for kw in [
            "free cash flow", "fcf", "operating cash", "capex", "capital expenditure",
            "cash from operations", "investing", "financing",
        ]):
            statement_type = "cash_flow"
        elif any(kw in combined for kw in [
            "total assets", "liabilities", "equity", "debt", "cash and cash equivalents",
            "goodwill", "working capital", "debt-to-equity", "balance sheet",
        ]):
            statement_type = "balance_sheet"

    if not statement_type:
        return {
            "chart_spec": None,
            "answer": "I couldn't determine which financial statement to use for this chart. Could you clarify — are you asking about the income statement, balance sheet, or cash flow?",
        }

    chart_type = state.get("chart_type") or "bar"

    try:
        # Step 1 — load DataFrame and fetch schema concurrently
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                df_future     = pool.submit(load_dataframes, companies, statement_type, years)
                schema_future = pool.submit(schema_retriever.get_schema, question, statement_type, 10)
                df         = df_future.result()
                schema_ctx = schema_future.result()
        except Exception:
            df         = load_dataframes(companies=companies, statement_type=statement_type, years=years)
            schema_ctx = schema_retriever.get_schema(question=question, statement_type=statement_type, top_k=10)

        # Step 2 — ask LLM for chart spec (schema already ready)
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

        # Step 5 — describe chart insights (llm_fast: just 2–3 sentences)
        resp   = (CHART_PROMPT | llm_fast).invoke({
            "data_summary": result_df.to_markdown(index=False)[:800],
            "question":     question,
        })

        # Use fig.to_json() → parse back to dict so all numpy types are
        # converted to native Python. Plain fig.to_dict() leaves ndarray /
        # np.int64 / np.float64 objects that crash json.dumps downstream.
        chart_spec = json.loads(fig.to_json())

        return {
            "chart_spec": chart_spec,
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
