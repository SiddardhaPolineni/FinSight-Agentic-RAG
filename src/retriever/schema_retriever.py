"""
FinSight Schema Retriever
──────────────────────────
Retrieves relevant column schema from Pinecone metadata namespace
based on the user's question.

Each record in the metadata namespace represents one financial column
with its description, aliases, unit and example values.

Usage:
    retriever = SchemaRetriever()
    schema_context = retriever.get_schema(
        question="what was Apple's revenue in 2024",
        statement_type="income_statement",
        top_k=8,
    )
"""

import json
import logging

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

import config as cfg

logger = logging.getLogger(__name__)

METADATA_NAMESPACE = "financial-schema"


class SchemaRetriever:
    """Retrieves relevant column schema from Pinecone for a given question."""

    def __init__(self):
        self._embedder = OpenAIEmbeddings(
            model=cfg.EMBEDDING_MODEL,
            openai_api_key=cfg.OPENAI_API_KEY,
        )
        pc = Pinecone(api_key=cfg.PINECONE_API_KEY)
        self._index = pc.Index(cfg.PINECONE_INDEX)
        logger.info("SchemaRetriever ready")

    def get_schema(
        self,
        question: str,
        statement_type: str,
        top_k: int = 10,
    ) -> str:
        """
        Retrieve the most relevant column definitions for the question
        and format them as a schema context string for the LLM.

        Args:
            question:       user question to match against column descriptions/aliases
            statement_type: "income_statement" | "balance_sheet" | "cash_flow"
            top_k:          number of column records to retrieve

        Returns:
            Formatted schema string ready to inject into the LLM prompt.
        """
        query_vector = self._embedder.embed_query(question)

        results = self._index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=METADATA_NAMESPACE,
            filter={"statement_type": {"$eq": statement_type}},
            include_metadata=True,
        )

        if not results.matches:
            logger.warning("No schema records found for statement_type=%s", statement_type)
            return f"No schema found for statement_type={statement_type}"

        index_cols_text = (
            "Index columns (always available for filtering):\n"
            "  company      (str)  — company name, e.g. 'Apple', 'Google', 'Microsoft', 'Nvidia'\n"
            "  fiscalYear   (str)  — 4-digit fiscal year, e.g. '2021', '2022', '2023', '2024', '2025'\n"
        )

        col_lines = []
        seen_cols = set()

        for match in results.matches:
            m = match.metadata
            col_name = m.get("column_name", "")
            if col_name in seen_cols:
                continue
            seen_cols.add(col_name)

            aliases = json.loads(m.get("aliases", "[]"))
            unit    = m.get("unit", "USD")
            desc    = m.get("description", "")
            alias_str = ", ".join(f'"{a}"' for a in aliases[:4])

            col_lines.append(
                f"  {col_name} ({unit})\n"
                f"    Description : {desc}\n"
                f"    Aliases     : [{alias_str}]"
            )

        return (
            f"Statement type: {statement_type}\n\n"
            f"{index_cols_text}\n"
            f"Relevant financial columns (ranked by relevance to query):\n"
            + "\n\n".join(col_lines)
        )

