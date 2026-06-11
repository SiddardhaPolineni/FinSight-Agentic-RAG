"""
FinSight — Pydantic schemas for FastAPI request / response payloads.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2)
    session_id: Optional[str] = Field(default=None)


class SourceDoc(BaseModel):
    source: str
    company: str
    year: str
    score: float
    snippet: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    chart_spec: Optional[dict]
    sources: list[SourceDoc]


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]
