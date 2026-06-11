# 📈 FinSight — Agentic Financial Research Assistant

Multi-agent RAG system over SEC 10-K filings + structured financial CSV data.

## Architecture

```
Streamlit UI ──► FastAPI ──► LangGraph
                                │
              ┌─────────────────┼──────────────────┐
              ▼                 ▼                   ▼
         rephrase_node    intent_node         (routes to)
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        csv_node           rag_node         chart_node
        (pandas)       (BM25+Pinecone        (plotly)
                        +Cohere rerank)
              └────────────────┼────────────────┘
                               ▼ (hybrid only)
                          synthesizer_node
                               ▼
                         ChatMessageHistory
```

## Data Coverage

| Type | Companies | Years |
|------|-----------|-------|
| Income Statement (CSV) | Apple, Google, Microsoft, Nvidia | FY2021–2025 |
| Balance Sheet (CSV) | Apple, Google, Microsoft, Nvidia | FY2021–2025 |
| Cash Flow (CSV) | Apple, Google, Microsoft, Nvidia | FY2021–2025 |
| SEC 10-K (PDF) | Google, Microsoft, NVIDIA | 2023, 2024, 2025 |

## Intent Routing

| Intent | Trigger | Action |
|--------|---------|--------|
| `csv_query` | Questions about specific financial metrics | Query CSV with pandas |
| `sec_rag` | Questions about strategy, risks, 10-K narrative | Hybrid BM25+Semantic+Rerank |
| `chart` | User asks for a plot/graph/chart | Pandas → Plotly chart |
| `hybrid` | Both structured data + 10-K context needed | CSV + RAG → Synthesizer |

## Setup

### 1. Create venv with UV

```bash
# Fix OneDrive hardlink issue first
set UV_LINK_MODE=copy

uv venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 3. Configure `.env`

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east-1
COHERE_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=FinSight-AI
```

### 4. Ingest SEC 10-K PDFs (one-time)

```bash
python -m src.ingestion
# Re-ingest from scratch:
python -m src.ingestion --reset
```

### 5. Start the backend

```bash
uvicorn src.api:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 6. Start Streamlit UI

```bash
streamlit run src/app.py
# App: http://localhost:8501
```

## Project Structure

```
FinSight-Agentic-RAG/
├── src/
│   ├── config.py        # all configuration (API keys, paths, model names)
│   ├── state.py         # LangGraph TypedDict state
│   ├── prompts.py       # all prompt templates
│   ├── ingestion.py     # PDF → Pinecone + BM25 corpus pipeline
│   ├── retriever.py     # BM25 + Pinecone + Cohere reranker
│   ├── csv_engine.py    # pandas CSV query engine
│   ├── nodes.py         # all 6 LangGraph nodes
│   ├── graph.py         # LangGraph workflow assembly
│   ├── api.py           # FastAPI backend + ChatMessageHistory
│   └── app.py           # Streamlit UI
├── data/
│   ├── Balance_Sheets/
│   ├── Income_Statments/
│   ├── Cash_Flow_Statments/
│   └── SEC_Filings/      ← place 10-K PDFs here
├── requirements.txt
├── .env                  ← never commit
└── .gitignore
```

## Example Questions

```
# CSV queries
"What was Nvidia's revenue in 2024?"
"Compare Apple and Microsoft net income from 2021 to 2025"
"Show me Google's free cash flow trend"
"What is Microsoft's total debt for 2023, 2024, 2025?"

# Charts
"Plot a bar chart of revenue for all companies in 2024"
"Show me a line chart of Nvidia's EPS from 2021 to 2025"

# SEC RAG
"What are NVIDIA's key risk factors in its 2025 10-K?"
"What did Microsoft say about AI strategy in 2024?"
"Summarize Google's business segments from the 2023 10-K"

# Hybrid
"Compare Nvidia's reported revenue with what they said about data center growth in 2024"
```
