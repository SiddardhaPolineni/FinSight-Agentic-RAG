"""
FinSight Hybrid Retriever
──────────────────────────
BM25 keyword search  +  Pinecone semantic search
  → Reciprocal Rank Fusion (RRF)
  → Cohere reranker

Returns a list of result dicts:
  {"content": str, "source": str, "company": str, "year": str, "score": float}
"""

import logging
import pickle
from typing import Optional

import cohere
import numpy as np
from rank_bm25 import BM25Okapi

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

import config as cfg

logger = logging.getLogger(__name__)

# ── In-memory BM25 state ──────────────────────────────────────────────────────
_bm25_corpus: list[dict] = []
_bm25_index: Optional[BM25Okapi] = None


def load_bm25_index() -> int:
    """Load BM25 corpus from disk into memory. Returns doc count."""
    global _bm25_corpus, _bm25_index
    if not cfg.BM25_CORPUS_PATH.exists():
        logger.warning("BM25 corpus not found at %s. Run src/ingestion.py first.", cfg.BM25_CORPUS_PATH)
        return 0
    with open(cfg.BM25_CORPUS_PATH, "rb") as f:
        _bm25_corpus = pickle.load(f)
    tokenized = [d["text"].lower().split() for d in _bm25_corpus]
    _bm25_index = BM25Okapi(tokenized)
    logger.info("BM25 index ready: %d documents", len(_bm25_corpus))
    return len(_bm25_corpus)


class HybridRetriever:
    def __init__(self) -> None:
        self._embeddings = OpenAIEmbeddings(
            model=cfg.EMBEDDING_MODEL,
            openai_api_key=cfg.OPENAI_API_KEY,
        )
        pc = Pinecone(api_key=cfg.PINECONE_API_KEY)
        index = pc.Index(cfg.PINECONE_INDEX)
        self._vector_store = PineconeVectorStore(
            index=index,
            embedding=self._embeddings,
            namespace=cfg.PINECONE_NAMESPACE,
        )
        self._cohere = cohere.Client(api_key=cfg.COHERE_API_KEY) if cfg.COHERE_API_KEY else None
        logger.info("HybridRetriever ready (reranker=%s)", "cohere" if self._cohere else "disabled")

    # ── Public ────────────────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        company_filter: Optional[list[str]] = None,
        year_filter: Optional[list[str]] = None,
        top_k: int = cfg.RETRIEVER_TOP_K,
        top_n: int = cfg.RERANKER_TOP_N,
    ) -> list[dict]:
        semantic = self._semantic(query, top_k)
        bm25     = self._bm25(query, top_k)
        fused    = self._rrf(semantic, bm25)

        # Metadata filtering
        if company_filter or year_filter:
            fused = [
                d for d in fused
                if (not company_filter or d["company"] in company_filter)
                and (not year_filter    or d["year"]    in year_filter)
            ]

        if self._cohere and fused:
            return self._rerank(query, fused, top_n)
        return fused[:top_n]

    # ── Private ───────────────────────────────────────────────────────────────
    def _semantic(self, query: str, k: int) -> list[tuple[dict, float]]:
        try:
            results = self._vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error("Semantic search failed: %s", e)
            return []
        out = []
        for doc, score in results:
            m = doc.metadata or {}
            out.append((
                {"content": doc.page_content, "source": m.get("source", ""),
                 "company": m.get("company", ""), "year": m.get("year", ""), "score": float(score)},
                float(score),
            ))
        return out

    def _bm25(self, query: str, k: int) -> list[tuple[dict, float]]:
        if _bm25_index is None or not _bm25_corpus:
            return []
        scores: np.ndarray = _bm25_index.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:k]
        out = []
        for idx in top_idx:
            s = float(scores[idx])
            if s <= 0:
                break
            m = _bm25_corpus[idx].get("meta", {})
            out.append((
                {"content": _bm25_corpus[idx]["text"], "source": m.get("source", ""),
                 "company": m.get("company", ""), "year": m.get("year", ""), "score": s},
                s,
            ))
        return out

    def _rrf(
        self,
        semantic: list[tuple[dict, float]],
        bm25: list[tuple[dict, float]],
        k: int = 60,
    ) -> list[dict]:
        scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        def key(d: dict) -> str:
            return f"{d['source']}::{d['content'][:150]}"

        for rank, (doc, _) in enumerate(semantic, 1):
            k_ = key(doc)
            scores[k_] = scores.get(k_, 0.0) + cfg.SEMANTIC_WEIGHT / (k + rank)
            doc_map[k_] = doc

        for rank, (doc, _) in enumerate(bm25, 1):
            k_ = key(doc)
            scores[k_] = scores.get(k_, 0.0) + cfg.BM25_WEIGHT / (k + rank)
            if k_ not in doc_map:
                doc_map[k_] = doc

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        fused = []
        for k_ in sorted_keys:
            d = doc_map[k_]
            d["score"] = round(scores[k_], 6)
            fused.append(d)
        return fused

    def _rerank(self, query: str, docs: list[dict], top_n: int) -> list[dict]:
        passages = [d["content"][:512] for d in docs]
        try:
            resp = self._cohere.rerank(
                model=cfg.COHERE_RERANK_MODEL,
                query=query,
                documents=passages,
                top_n=top_n,
            )
            reranked = []
            for r in resp.results:
                d = docs[r.index]
                d["score"] = round(r.relevance_score, 6)
                reranked.append(d)
            return reranked
        except Exception as e:
            logger.error("Cohere rerank failed: %s — using RRF order", e)
            return docs[:top_n]
