"""
All prompt templates for FinSight nodes.
"""

from langchain_core.prompts import ChatPromptTemplate

# ── 1. Analyze — rephrase + intent in one call ────────────────────────────────
ANALYZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Financial query classifier. Return JSON only — no fences, no explanation.

{{
  "rephrased_query": "<self-contained rewrite>",
  "intent": "<csv_query|sec_rag|chart|hybrid>",
  "companies": ["apple|google|microsoft|nvidia"],
  "years": ["2021-2025 as strings"],
  "metrics": ["metric names mentioned in the question"],
  "chart_type": "<bar|line|pie|null>"
}}

Intent:
- csv_query  → financial figures, ratios, margins, KPIs from statements
- sec_rag    → strategy, risks, MD&A, qualitative info from 10-K
- chart      → user wants a plot/chart/visualization
- hybrid     → needs both figures AND 10-K narrative

Tickers: AAPL→apple, NVDA→nvidia, MSFT→microsoft, GOOG/GOOGL/Alphabet→google

IMPORTANT for years:
- Only include years the user EXPLICITLY mentions
- If no year specified, return empty []
- "all years", "trend", "over the years" → return []

History:
{history}"""),
    ("human", "Question: {question}"),
])

# ── 3. CSV pandas query builder ────────────────────────────────────────────────
PANDAS_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a pandas + financial analyst expert. DataFrame `df` schema:

{column_context}

Return a JSON object with exactly these keys:
{{
  "expr": "<pandas expression that computes the answer>",
  "answer_template": "<natural-language sentence with {{result}} as placeholder>"
}}

Rules for expr:
- Use ONLY column names from the schema
- Filter with a combined boolean mask:
    df[(df["company"] == "Microsoft") & (df["fiscalYear"] == "2024")]["revenue"].iloc[0]
- For derived metrics compute inline:
    net profit margin  → netIncome / revenue
    operating margin   → operatingIncome / revenue
    gross margin       → grossProfit / revenue
    ROA                → netIncome / totalAssets (need balance_sheet)
    debt-to-equity     → totalDebt / totalStockholdersEquity
- For trends/comparisons return a small DataFrame or Series
- No imports, no print

Rules for answer_template:
- Write a clear 1–3 sentence answer as if the {{result}} value is known
- Use {{result}} where the computed value goes
- Include company names, fiscal years, and metric context
- Format hints: if monetary use "$" prefix, if ratio/margin use "%" suffix

Examples:
  Question: "What was Apple's revenue in 2024?"
  → {{"expr": "df[(df[\\"company\\"] == \\"Apple\\") & (df[\\"fiscalYear\\"] == \\"2024\\")]['revenue'].iloc[0]", "answer_template": "Apple's revenue in fiscal year 2024 was ${{result}}."}}

  Question: "Compare Google and Microsoft net income for 2023"
  → {{"expr": "df[df[\\"fiscalYear\\"] == \\"2023\\"][[\\"company\\",\\"fiscalYear\\",\\"netIncome\\"]]", "answer_template": "Here is the net income comparison for FY 2023:\\n{{result}}"}}

Return ONLY the JSON — no markdown fences."""),
    ("human", "Question: {question}\n\nJSON:"),
])

# ── 3b. Chart pandas query + axis spec ─────────────────────────────────────────
CHART_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a pandas expert and chart designer working with financial data.

You have a DataFrame called `df`. Here is its schema with real sample values:

{column_context}

Return a JSON object with exactly these keys:
{{
  "expr":  "<pandas expression returning a DataFrame ready to plot>",
  "x":     "<column name for x-axis>",
  "y":     "<column name for y-axis>",
  "color": "<column name for color/grouping, or null if single series>",
  "title": "<short descriptive chart title>"
}}

Rules:
- Use ONLY column names shown in the schema above
- expr MUST return a DataFrame — never a scalar or raw Series
- Always use a combined boolean mask for filtering
- Do NOT use print(), imports, or plt

Axis selection:
  | Question type                               | x          | color      |
  |---------------------------------------------|------------|------------|
  | Trend over time, one company                | fiscalYear | null       |
  | Trend over time, multiple companies         | fiscalYear | company    |
  | Compare companies at one point in time      | company    | null       |

Return ONLY the JSON — no markdown fences."""),
    ("human", "Question: {question}\n\nChart spec JSON:"),
])

# ── 4. CSV result interpreter ───────────────────────────────────────────────────
CSV_ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial assistant. A pandas query has already computed
the exact answer to the user's question. Your only job is to present that
computed result as a clear, concise sentence or short paragraph.

Rules:
- Do NOT re-analyse or re-calculate — the result is already correct
- Format numbers clearly (e.g. $130.50B, 42.3%, 6.11 EPS)
- Mention the company name and fiscal year where relevant
- Keep it brief — 1 to 4 sentences maximum"""),
    ("human", """Question: {question}

Computed result:
{data}

Present this as a natural-language answer:"""),
])

# ── 4. SEC RAG analyst ─────────────────────────────────────────────────────────
SEC_ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are FinSight, a financial analyst specializing in SEC 10-K filings.
Answer ONLY using the provided document excerpts.
- Reference the source (company name, year) inline when relevant
- Be precise about risks, strategies, and business segments
- Do NOT fabricate facts not present in the context"""),
    ("human", """Context Documents:
{context}

Question: {question}

Provide a well-cited answer:"""),
])

# ── 5. Chart describer ─────────────────────────────────────────────────────────
CHART_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial data visualization expert.
The user wants a chart. Describe what the chart shows in 2-3 sentences after the chart is rendered.
Be specific about trends, peaks, and notable comparisons visible in the data."""),
    ("human", """Chart data summary:
{data_summary}

Question: {question}

Describe the key insights from this chart:"""),
])

# ── 6. Synthesizer ─────────────────────────────────────────────────────────────
SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are FinSight's response formatter.
Combine the analysis results into a single, clean, well-structured response.
- Preserve all numbers and citations
- Use markdown formatting (headers, bullet points, bold for numbers)
- End with a one-line disclaimer: *Analysis based on public financial data. Not investment advice.*
- Do NOT add information not present in the inputs."""),
    ("human", """Original question: {question}

CSV Analysis:
{csv_result}

SEC Filing Analysis:
{rag_result}

Produce the final response:"""),
])
