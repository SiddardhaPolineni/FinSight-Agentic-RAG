"""
FinSight Metadata Ingestion
────────────────────────────
Reads the three financial statement metadata JSON files and upserts
each column's schema entry into Pinecone as a searchable vector.

Each Pinecone record represents ONE column from ONE statement type:
  - text embedded: "{column_name}: {description}. Aliases: {aliases}"
  - metadata stored: statement_type, column_name, description, aliases, unit, examples

Run:
    python ingest_metadata.py

The metadata namespace is separate from the SEC 10-K namespace so
there is no interference between document retrieval and schema retrieval.
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


def build_embed_text(col_name: str, col_meta: dict) -> str:
    """
    Build the text string to embed for a single column.
    Combines column name, description and aliases so semantic search
    can match natural-language queries to the right column.
    """
    aliases = ", ".join(col_meta.get("aliases", []))
    description = col_meta.get("description", "")
    return f"{col_name}: {description} Aliases: {aliases}"


def build_pinecone_record(
    col_name: str,
    col_meta: dict,
    statement_type: str,
    embedding: list[float],
) -> dict:
    """Build a Pinecone upsert record for a single column."""
    return {
        "id":     f"{statement_type}__{col_name}",
        "values": embedding,
        "metadata": {
            "statement_type": statement_type,
            "column_name":    col_name,
            "description":    col_meta.get("description", ""),
            "aliases":        json.dumps(col_meta.get("aliases", [])),
            "unit":           col_meta.get("unit", "USD"),
            "examples":       json.dumps(col_meta.get("examples", [])),
            "embed_text":     build_embed_text(col_name, col_meta),
        },
    }


def ingest_metadata():
    logger.info("="*60)
    logger.info("FinSight Metadata Ingestion")
    logger.info("Index     : %s", cfg.PINECONE_INDEX)
    logger.info("Namespace : %s", METADATA_NAMESPACE)
    logger.info("="*60)

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

    for statement_type, path in METADATA_FILES.items():
        if not path.exists():
            logger.error("Metadata file not found: %s", path)
            continue

        with open(path, encoding="utf-8") as f:
            metadata = json.load(f)

        columns = metadata.get("columns", {})
        logger.info("Processing %s — %d columns", statement_type, len(columns))

        # Build texts to embed
        texts   = [build_embed_text(col, meta) for col, meta in columns.items()]
        col_names = list(columns.keys())

        # Embed all columns for this statement in one batch
        embeddings = embedder.embed_documents(texts)

        # Build and upsert records
        records = [
            build_pinecone_record(col_names[i], columns[col_names[i]], statement_type, embeddings[i])
            for i in range(len(col_names))
        ]

        index.upsert(vectors=records, namespace=METADATA_NAMESPACE)
        logger.info("  Upserted %d column records for %s", len(records), statement_type)
        total_upserted += len(records)

    logger.info("="*60)
    logger.info("Done. Total records upserted: %d", total_upserted)
    logger.info("="*60)


if __name__ == "__main__":
    ingest_metadata()
