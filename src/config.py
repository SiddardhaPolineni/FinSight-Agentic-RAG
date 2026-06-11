"""
FinSight — Central Configuration
All static settings live here. Secrets come from environment / .env.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root
ROOT_DIR   = Path(__file__).parent.parent
DATA_DIR   = ROOT_DIR / "data"
SRC_DIR    = ROOT_DIR / "src"

# OpenAI
OPENAI_API_KEY   : str   = os.environ["OPENAI_API_KEY"]
LLM_MODEL        : str   = "gpt-4o-mini"
LLM_TEMPERATURE  : float = 0.0
LLM_MAX_TOKENS   : int   = 2048
EMBEDDING_MODEL  : str   = "text-embedding-3-small"
EMBEDDING_DIM    : int   = 1536

# Pinecone
PINECONE_API_KEY    : str = os.environ["PINECONE_API_KEY"]
PINECONE_ENV        : str = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX      : str = "finsight-sec"
PINECONE_NAMESPACE  : str = "sec-10k"

# Cohere reranker
COHERE_API_KEY      : str = os.environ.get("COHERE_API_KEY", "")
COHERE_RERANK_MODEL : str = "rerank-english-v3.0"

# LangSmith tracing
LANGCHAIN_TRACING_V2 : str = os.environ.get("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY    : str = os.environ.get("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT    : str = os.environ.get("LANGCHAIN_PROJECT", "FinSight-AI")


os.environ.setdefault("LANGCHAIN_TRACING_V2", LANGCHAIN_TRACING_V2)
os.environ.setdefault("LANGCHAIN_API_KEY",    LANGCHAIN_API_KEY)
os.environ.setdefault("LANGCHAIN_PROJECT",    LANGCHAIN_PROJECT)

# Retrieval
RETRIEVER_TOP_K  : int   = 10
RERANKER_TOP_N   : int   = 5
BM25_WEIGHT      : float = 0.4
SEMANTIC_WEIGHT  : float = 0.6

# Ingestion
CHUNK_SIZE      : int = 1000
CHUNK_OVERLAP   : int = 200
MIN_CHUNK_CHARS : int = 80
BM25_CORPUS_PATH = DATA_DIR / "bm25_corpus.pkl"

# CSV data paths
CSV_PATHS = {
    "balance_sheet": {
        "apple":     DATA_DIR/"Balance_Sheets"/"Apple.csv",
        "google":    DATA_DIR/"Balance_Sheets"/"Google.csv",
        "microsoft": DATA_DIR/"Balance_Sheets"/"Microsoft.csv",
        "nvidia":    DATA_DIR/"Balance_Sheets"/"Nvidia.csv",
    },
    "income_statement": {
        "apple":     DATA_DIR/"Income_Statments"/"Apple.csv",
        "google":    DATA_DIR/"Income_Statments"/"Google.csv",
        "microsoft": DATA_DIR/"Income_Statments"/"Microsoft.csv",
        "nvidia":    DATA_DIR/"Income_Statments"/"Nvidia.csv",
    },
    "cash_flow": {
        "apple":     DATA_DIR/"Cash_Flow_Statments"/"Apple.csv",
        "google":    DATA_DIR/"Cash_Flow_Statments"/"Google.csv",
        "microsoft": DATA_DIR/"Cash_Flow_Statments"/"Microsoft.csv",
        "nvidia":    DATA_DIR/"Cash_Flow_Statments"/"Nvidia.csv",
    },
}

# SEC filing PDFs
SEC_DIR = DATA_DIR / "SEC_Filings"

# Companies that have CSV data
SUPPORTED_COMPANIES = ["apple", "google", "microsoft", "nvidia"]

# Companies with SEC 10-K PDFs
SEC_COMPANIES = ["apple", "google", "microsoft", "nvidia"]

# ── API server ────────────────────────────────────────────────────────────────
API_HOST : str = "0.0.0.0"
API_PORT : int = 8000

# ── Streamlit ─────────────────────────────────────────────────────────────────
APP_TITLE    : str = "FinSight - Financial Research Assistant"
APP_SUBTITLE : str = "Ask questions about financials, SEC filings, and visualize the data."
