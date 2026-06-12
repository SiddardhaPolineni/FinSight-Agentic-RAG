"""
FinSight Schema Retriever
──────────────────────────
Retrieves relevant column schema AND KPI definitions from Pinecone
based on the user's question.

Key behavior:
  - Searches across ALL statement types (no pre-filtering)
  - Infers statement_type from the retrieved results
  - Returns both schema context (for the LLM) and statement_type (for DataFrame loading)
"""

import json
import logging
from collections import Counter

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

import config as cfg

logger = logging.getLogger(__name__)

METADATA_NAMESPACE = "financial-schema"


class SchemaRetriever:
    """Retrieves relevant columns + KPIs from Pinecone for a given question."""

    def __init__(self):
        self._embedder = OpenAIEmbeddings(
            model=cfg.EMBEDDING_MODEL,
            openai_api_key=cfg.OPENAI_API_KEY,
        )
        pc = Pinecone(api_key=cfg.PINECONE_API_KEY)
        self._index = pc.Index(cfg.PINECONE_INDEX)
        self._cache: dict[str, dict] = {}
        logger.info("SchemaRetriever ready")

    def retrieve(self, question: str, top_k: int = 10) -> dict:
        """
        Retrieve relevant columns and KPIs for a question.

        Args:
            question: the user's financial question
            top_k: number of records to retrieve from Pinecone

        Returns:
            {
                "statement_type": "income_statement" (or comma-separated for cross-statement),
                "schema_context": "formatted string for the LLM prompt",
                "kpis": [{"name": ..., "formula": ..., ...}]
            }
        """
        cache_key = question.strip().lower()
        if cache_key in self._cache:
            return self._cache[cache_key]

        query_vector = self._embedder.embed_query(question)

        # Search across ALL statement types
        results = self._index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=METADATA_NAMESPACE,
            include_metadata=True,
        )

        if not results.matches:
            logger.warning("No schema records found for query: %s", question[:60])
            result = {
                "statement_type": None,
                "schema_context": "No relevant schema found.",
                "kpis": [],
            }
            self._cache[cache_key] = result
            return result

        # Separate columns and KPIs from results
        columns = []
        kpis = []
        statement_types = []
        top_statement_type = None  # from the highest-ranked result

        for match in results.matches:
            m = match.metadata
            record_type = m.get("type", "column")

            if record_type == "kpi":
                kpis.append({
                    "name":     m.get("kpi_name", ""),
                    "formula":  m.get("formula", ""),
                    "required_columns": json.loads(m.get("required_columns", "[]")),
                    "statement_type":   m.get("statement_type", ""),
                    "description":      m.get("description", ""),
                })
                for st in m.get("statement_type", "").split(","):
                    st = st.strip()
                    if st:
                        statement_types.append(st)
                        if top_statement_type is None:
                            top_statement_type = st
            else:
                col_name = m.get("column_name", "")
                stmt     = m.get("statement_type", "")
                if col_name not in [c["column_name"] for c in columns]:
                    columns.append({
                        "column_name":    col_name,
                        "statement_type": stmt,
                        "description":    m.get("description", ""),
                        "aliases":        json.loads(m.get("aliases", "[]")),
                        "unit":           m.get("unit", "USD"),
                    })
                if stmt:
                    statement_types.append(stmt)
                    if top_statement_type is None:
                        top_statement_type = stmt

        # Use the top-ranked result's statement_type (most semantically relevant)
        inferred_type = top_statement_type or self._infer_statement_type(statement_types)

        # Filter columns to only those from the inferred statement type
        relevant_columns = [c for c in columns if c["statement_type"] == inferred_type]
        # If no columns match (e.g. only KPIs returned), keep all
        if not relevant_columns:
            relevant_columns = columns

        # Build schema context string for the LLM
        schema_context = self._format_schema_context(relevant_columns, kpis, inferred_type)

        result = {
            "statement_type": inferred_type,
            "schema_context": schema_context,
            "kpis": kpis,
        }
        self._cache[cache_key] = result
        return result

    # ── Keep backward compatibility ──────────────────────────────────────────
    def get_schema(self, question: str, statement_type: str = None, top_k: int = 10) -> str:
        """Backward-compatible method. Returns just the schema context string."""
        result = self.retrieve(question, top_k)
        return result["schema_context"]


    def _infer_statement_type(self, statement_types: list[str]) -> str:
        """Pick the most frequent statement_type from retrieved results."""
        if not statement_types:
            return "income_statement"  # safe default
        counts = Counter(statement_types)
        # Return the most common one
        return counts.most_common(1)[0][0]

    def _format_schema_context(self, columns: list[dict], kpis: list[dict], statement_type: str) -> str:
        """Format columns and KPIs into a readable context string for the LLM."""
        index_cols = (
            "Index columns (always available):\n"
            "  company      (str)  — 'Apple', 'Google', 'Microsoft', 'Nvidia'\n"
            "  fiscalYear   (str)  — '2021', '2022', '2023', '2024', '2025'\n"
        )

        col_lines = []
        for col in columns:
            aliases = ", ".join(f'"{a}"' for a in col["aliases"][:4])
            col_lines.append(
                f"  {col['column_name']} ({col['unit']})\n"
                f"    {col['description']}\n"
                f"    Aliases: [{aliases}]"
            )

        kpi_lines = []
        for kpi in kpis:
            kpi_lines.append(
                f"  {kpi['name']} = {kpi['formula']}\n"
                f"    {kpi['description']}"
            )

        parts = [f"Primary statement: {statement_type}\n", index_cols]

        if col_lines:
            parts.append("\nRelevant columns:\n" + "\n\n".join(col_lines))

        if kpi_lines:
            parts.append("\nRelevant KPIs (use these formulas):\n" + "\n\n".join(kpi_lines))

        return "\n".join(parts)
