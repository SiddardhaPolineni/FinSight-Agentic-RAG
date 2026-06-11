"""
FinSight Metadata Ingestion
────────────────────────────
Ingests column schema AND KPI definitions into Pinecone.

Records:
  - Column records: one per column per statement_type
  - KPI records: one per derived metric (net margin, ROA, etc.)

Embedded text includes descriptions, aliases, and derived metric hints
so that semantic search reliably matches user queries to the right columns.

Run:
    python ingest_metadata.py
"""

import json
import logging
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

METADATA_DIR       = cfg.DATA_DIR / "metadata"
METADATA_NAMESPACE = "financial-schema"

METADATA_FILES = {
    "income_statement": METADATA_DIR / "income_statement.json",
    "balance_sheet":    METADATA_DIR / "balance_sheet.json",
    "cash_flow":        METADATA_DIR / "cash_flow.json",
}

KPIS_FILE = METADATA_DIR / "kpis.json"


def build_column_embed_text(col_name: str, col_meta: dict) -> str:
    """
    Build embed text for a column record.
    Includes name, description, and all aliases for broad semantic matching.
    """
    aliases = ", ".join(col_meta.get("aliases", []))
    description = col_meta.get("description", "")
    return f"{col_name}: {description} Aliases: {aliases}"


def build_kpi_embed_text(kpi_name: str, kpi_meta: dict) -> str:
    """
    Build embed text for a KPI record.
    Includes name, description, formula, and aliases so that
    questions like "profit margin" or "revenue growth" match correctly.
    """
    aliases = ", ".join(kpi_meta.get("aliases", []))
    description = kpi_meta.get("description", "")
    formula = kpi_meta.get("formula", "")
    return f"{kpi_name}: {description} Formula: {formula}. Aliases: {aliases}"


def ingest_metadata():
    logger.info("=" * 60)
    logger.info("FinSight Metadata Ingestion")
    logger.info("Index     : %s", cfg.PINECONE_INDEX)
    logger.info("Namespace : %s", METADATA_NAMESPACE)
    logger.info("=" * 60)

    # Init Pinecone
    pc = Pinecone(api_key=cfg.PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]
    if cfg.PINECONE_INDEX not in existing:
        logger.info("Creating Pinecone index '%s'...", cfg.PINECONE_INDEX)
        pc.create_index(
            name=cfg.PINECONE_INDEX,
            dimension=cfg.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=cfg.PINECONE_ENV),
        )
    index = pc.Index(cfg.PINECONE_INDEX)

    # Init embeddings
    embedder = OpenAIEmbeddings(
        model=cfg.EMBEDDING_MODEL,
        openai_api_key=cfg.OPENAI_API_KEY,
    )

    total_upserted = 0

    # ── Ingest column records ─────────────────────────────────────────────────
    for statement_type, path in METADATA_FILES.items():
        if not path.exists():
            logger.error("Metadata file not found: %s", path)
            continue

        with open(path, encoding="utf-8") as f:
            metadata = json.load(f)

        columns = metadata.get("columns", {})
        logger.info("Processing %s — %d columns", statement_type, len(columns))

        texts = []
        records = []

        for col_name, col_meta in columns.items():
            embed_text = build_column_embed_text(col_name, col_meta)
            texts.append(embed_text)
            records.append({
                "id": f"{statement_type}__{col_name}",
                "metadata": {
                    "type":           "column",
                    "statement_type": statement_type,
                    "column_name":    col_name,
                    "description":    col_meta.get("description", ""),
                    "aliases":        json.dumps(col_meta.get("aliases", [])),
                    "unit":           col_meta.get("unit", "USD"),
                    "embed_text":     embed_text,
                },
            })

        # Embed all at once
        embeddings = embedder.embed_documents(texts)
        for i, rec in enumerate(records):
            rec["values"] = embeddings[i]

        index.upsert(vectors=records, namespace=METADATA_NAMESPACE)
        logger.info("  Upserted %d column records for %s", len(records), statement_type)
        total_upserted += len(records)

    # ── Ingest KPI records ────────────────────────────────────────────────────
    if KPIS_FILE.exists():
        with open(KPIS_FILE, encoding="utf-8") as f:
            kpis_data = json.load(f)

        kpis = kpis_data.get("kpis", {})
        logger.info("Processing KPIs — %d records", len(kpis))

        texts = []
        records = []

        for kpi_name, kpi_meta in kpis.items():
            embed_text = build_kpi_embed_text(kpi_name, kpi_meta)
            texts.append(embed_text)
            records.append({
                "id": f"kpi__{kpi_name}",
                "metadata": {
                    "type":             "kpi",
                    "kpi_name":         kpi_name,
                    "formula":          kpi_meta.get("formula", ""),
                    "required_columns": json.dumps(kpi_meta.get("required_columns", [])),
                    "statement_type":   kpi_meta.get("statement_type", ""),
                    "description":      kpi_meta.get("description", ""),
                    "aliases":          json.dumps(kpi_meta.get("aliases", [])),
                    "embed_text":       embed_text,
                },
            })

        embeddings = embedder.embed_documents(texts)
        for i, rec in enumerate(records):
            rec["values"] = embeddings[i]

        index.upsert(vectors=records, namespace=METADATA_NAMESPACE)
        logger.info("  Upserted %d KPI records", len(records))
        total_upserted += len(records)
    else:
        logger.warning("KPIs file not found: %s", KPIS_FILE)

    logger.info("=" * 60)
    logger.info("Done. Total records upserted: %d", total_upserted)
    logger.info("=" * 60)


if __name__ == "__main__":
    ingest_metadata()
