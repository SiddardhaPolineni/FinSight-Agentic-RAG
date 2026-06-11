"""
FinSight Ingestion Pipeline
────────────────────────────
Standalone pipeline — run this script directly whenever you want to
(re)ingest the SEC 10-K PDFs into Pinecone and rebuild the BM25 corpus.

What it does:
  1. Scans data/SEC_Filings/ for all .pdf files
  2. Creates the Pinecone index if it doesn't exist
  3. Loads each PDF page-by-page (PyPDFLoader)
  4. Splits pages into overlapping text chunks
  5. Embeds chunks with OpenAI text-embedding-3-small
  6. Upserts chunks into Pinecone (batches of 100)
  7. Saves a BM25 corpus pickle alongside for hybrid keyword search

Run:
    python -m src.ingestion.ingestion
"""

import logging
import pickle
import re
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Organization name detection from filename
ORG_MAP = {
    "googl": "google",
    "google":    "google",
    "alphabet":  "google",
    "microsoft": "microsoft",
    "msft":      "microsoft",
    "nvidia":    "nvidia",
    "nvda":      "nvidia",
    "apple":     "apple",
    "aapl":      "apple",
}


def parse_filename(filename: str) -> dict:
    """
    Extract company and fiscal year from a filename.
    e.g. 'NVIDIA-10-K-2024.pdf' → {company: 'nvidia', year: '2024', ...}
    """
    lower = filename.lower()
    company = next((v for k, v in ORG_MAP.items() if k in lower), "unknown")
    match = re.search(r"(20\d{2})", filename)
    year = match.group(1) if match else "unknown"
    return {
        "company":  company,
        "year":     year,
        "source":   filename,
        "doc_type": "10-K",
    }


# verify if pinecone index is avilable

def verify_index(pc: Pinecone) -> None:
    """Create the Pinecone index if it does not already exist."""
    
    existing_indices = [idx.name for idx in pc.list_indexes()]
    if cfg.PINECONE_INDEX not in existing_indices:
        logger.info("Creating Pinecone index '%s' (dim=%d)…", cfg.PINECONE_INDEX, cfg.EMBEDDING_DIM)
        pc.create_index(
            name=cfg.PINECONE_INDEX,
            dimension=cfg.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=cfg.PINECONE_ENV),
        )
        logger.info("Index created.")
    else:
        logger.info("Index '%s' already exists — skipping creation.", cfg.PINECONE_INDEX)


# load BM25 corpus

def load_bm25_corpus() -> list[dict]:
    """Load existing BM25 corpus from disk (empty list if not found)."""
    if cfg.BM25_CORPUS_PATH.exists():
        with open(cfg.BM25_CORPUS_PATH, "rb") as f:
            corpus = pickle.load(f)
        logger.info("Loaded existing BM25 corpus (%d docs)", len(corpus))
        return corpus
    logger.info("No existing BM25 corpus found — starting fresh.")
    return []


def save_bm25_corpus(corpus: list[dict]) -> None:
    """Persist BM25 corpus to disk."""
    cfg.BM25_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.BM25_CORPUS_PATH, "wb") as f:
        pickle.dump(corpus, f)
    logger.info("BM25 corpus saved → %s  (%d total docs)", cfg.BM25_CORPUS_PATH, len(corpus))


# Per sec file ingestion

def ingest_sec(pdf_path: Path, pc: Pinecone, embeddings: OpenAIEmbeddings, corpus: list[dict]) -> list[dict]:
    """
    Full pipeline for one SEC PDF:
      load pages → chunk → upsert Pinecone → append BM25 corpus
    Returns the updated corpus list.
    """
    meta = parse_filename(pdf_path.name)
    logger.info("━━━ %s  (company=%s, year=%s)", pdf_path.name, meta["company"], meta["year"])

    # Step 1 — Load PDF pages
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    for page in pages:
        page.metadata.update(meta)
    logger.info("  Loaded %d pages", len(pages))

    # Step 2 — Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= cfg.MIN_CHUNK_CHARS]
    logger.info("  %d chunks after filtering (min_chars=%d)", len(chunks), cfg.MIN_CHUNK_CHARS)

    if not chunks:
        logger.warning("  No usable chunks extracted from %s — skipping.", pdf_path.name)
        return corpus

    # Step 3 — Embed and upsert to Pinecone in batches
    batch_size = 100
    total_batches = (len(chunks) - 1) // batch_size + 1
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=cfg.PINECONE_INDEX,
            namespace=cfg.PINECONE_NAMESPACE,
        )
        
        logger.info("  Batch %d/%d — upserted %d chunks",i // batch_size + 1, total_batches, len(batch))
    
    logger.info("  ✓ Pinecone upsert complete (%d total chunks)", len(chunks))

    # Step 4 — Append to BM25 corpus
    for chunk in chunks:
        corpus.append({"text": chunk.page_content, "meta": chunk.metadata})
    logger.info("  ✓ BM25 corpus updated (+%d docs)", len(chunks))

    return corpus


# ingestion pipeline

def ingestion_pipeline() -> None:
    """
    Scan data/SEC_Filings/, ingest every PDF into Pinecone, and save
    the BM25 corpus.  Always does a fresh ingest of all files found.
    If a file was already ingested, its vectors will be overwritten
    (Pinecone upsert is idempotent by content hash).
    """
    logger.info("="*10)
    logger.info("  FinSight Ingestion Pipeline")
    logger.info("  SEC_DIR : %s", cfg.SEC_DIR)
    logger.info("  Index   : %s / namespace=%s", cfg.PINECONE_INDEX, cfg.PINECONE_NAMESPACE)
    logger.info("="*10)

    # Discover PDFs
    pdfs = sorted(cfg.SEC_DIR.glob("*.pdf"))
    if not pdfs:
        logger.error("No PDF files found in %s", cfg.SEC_DIR)
        logger.error("Place your 10-K PDFs there and re-run.")
        return

    logger.info("Found %d PDF(s):", len(pdfs))
    for p in pdfs:
        logger.info("  • %s", p.name)

    # Initialise clients
    pc = Pinecone(api_key=cfg.PINECONE_API_KEY)
    verify_index(pc)

    embeddings = OpenAIEmbeddings(
        model=cfg.EMBEDDING_MODEL,
        openai_api_key=cfg.OPENAI_API_KEY,
    )

    # Load existing BM25 corpus (so we can append rather than replace)
    corpus = load_bm25_corpus()
    start_count = len(corpus)

    # Ingest each PDF
    for pdf in pdfs:
        try:
            corpus = ingest_sec(pdf, pc, embeddings, corpus)
        except Exception as exc:
            logger.error(" Failed to ingest %s: %s", pdf.name, exc, exc_info=True)

    # Save updated corpus
    save_bm25_corpus(corpus)

    logger.info(10*"=")
    logger.info("  Ingestion complete!")
    logger.info("  PDFs processed      : %d", len(pdfs))
    logger.info("  New BM25 chunks     : %d", len(corpus) - start_count)
    logger.info("  Total BM25 corpus   : %d", len(corpus))
    logger.info(10*"=")

if __name__ == "__main__":
    ingestion_pipeline()
