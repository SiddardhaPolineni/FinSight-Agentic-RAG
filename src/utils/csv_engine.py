"""
FinSight CSV Engine
────────────────────
Loads financial CSVs into a DataFrame and provides column context
(actual column names + sample values) for the LLM to write pandas queries.

CSV format note: source CSVs have one extra leading column (the date,
a row-index from the original export) not reflected in the header row.
Apple income statement is the exception — it has no extra column.
We detect this by comparing header count vs data column count.
"""

import csv
import logging
from typing import Optional

import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)

# Columns that are index/metadata — not financial metrics
_META_COLS = frozenset({
    "date", "symbol", "reportedCurrency", "cik",
    "filingDate", "acceptedDate", "fiscalYear", "period", "company",
})

# Keywords that map to each statement type — used for inference when LLM returns null
_STATEMENT_KEYWORDS = {
    "income_statement": [
        "revenue", "sales", "income", "profit", "loss", "earnings", "eps",
        "ebitda", "ebit", "margin", "gross", "operating", "tax", "interest",
    ],
    "balance_sheet": [
        "assets", "liabilities", "equity", "debt", "cash", "inventory",
        "receivable", "payable", "goodwill", "investment", "capital",
    ],
    "cash_flow": [
        "cash flow", "capex", "free cash", "operating cash", "buyback",
        "dividend", "financing", "investing", "depreciation", "working capital",
    ],
}


def infer_statement_type(question: str) -> str:
    """Infer statement type from question keywords. Returns income_statement as default."""
    q = question.lower()
    scores = {st: 0 for st in _STATEMENT_KEYWORDS}
    for st, keywords in _STATEMENT_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                scores[st] += 1
    best = max(scores, key=scores.get)
    # Default to income_statement when scores are tied or all zero
    return best if scores[best] > 0 else "income_statement"


def sanitize_statement_type(raw) -> Optional[str]:
    """Return None if raw is a null-like or invalid statement type string."""
    if raw is None:
        return None
    s = str(raw).lower().strip()
    if s in ("null", "none", "n/a", "", "undefined"):
        return None
    valid = {"income_statement", "balance_sheet", "cash_flow"}
    return raw if s in valid else None


def _read_csv_fixed(path) -> pd.DataFrame:
    """
    Read a financial CSV handling two formats:
      - diff=1: data rows have one extra leading value (date) not in header
      - diff=0: standard CSV where 'date' is already the first header column

    Returns a DataFrame with numeric columns cast to float and
    'fiscalYear' set to a 4-digit year string.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    if not rows:
        return pd.DataFrame(columns=headers)

    n_headers = len(headers)
    n_cols    = len(rows[0])

    if n_cols == n_headers + 1:
        # Extra leading column is the date
        date_col = [r[0] for r in rows]
        data     = [r[1:n_headers + 1] for r in rows]
        df = pd.DataFrame(data, columns=headers)
        df["date"]       = date_col
        df["fiscalYear"] = (
            pd.to_datetime(pd.Series(date_col), errors="coerce")
            .dt.year.astype("Int64").astype(str)
            .replace("<NA>", "")
        )
    else:
        # Standard format — date is already a named column
        data = [r[:n_headers] for r in rows]
        df = pd.DataFrame(data, columns=headers)
        date_src = df["date"] if "date" in df.columns else df.get("fiscalYear", pd.Series())
        df["fiscalYear"] = (
            pd.to_datetime(date_src, errors="coerce")
            .dt.year.astype("Int64").astype(str)
            .replace("<NA>", "")
        )

    # Cast all non-meta columns to numeric
    for col in df.columns:
        if col not in _META_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_dataframes(
    companies: list[str],
    statement_type: str,
    years: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load CSVs for the given companies and statement type and return
    a single combined DataFrame.

    Args:
        companies:      lowercase company names e.g. ["apple", "nvidia"]
        statement_type: "income_statement" | "balance_sheet" | "cash_flow"
        years:          optional year filter e.g. ["2023", "2024"]

    Returns:
        combined DataFrame

    Raises:
        ValueError: if statement_type is invalid or no data found.
    """
    valid_types = {"income_statement", "balance_sheet", "cash_flow"}
    if statement_type not in valid_types:
        raise ValueError(
            f"Invalid statement_type '{statement_type}'. Must be one of: {valid_types}"
        )

    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for company in companies:
        path = cfg.CSV_PATHS.get(statement_type, {}).get(company.lower())
        if path is None or not path.exists():
            logger.warning("CSV not found — company=%s, statement=%s", company, statement_type)
            missing.append(company)
            continue

        df = _read_csv_fixed(path)
        df["company"] = company.title()

        if years:
            df = df[df["fiscalYear"].isin(years)]

        frames.append(df)
        logger.info(
            "Loaded %s / %s — %d rows | years: %s",
            company, statement_type, len(df), df["fiscalYear"].unique().tolist(),
        )

    if not frames:
        raise ValueError(
            f"No CSV data found for companies={companies}, "
            f"statement={statement_type}. Missing: {missing}"
        )

    if missing:
        logger.warning("Could not load data for: %s", missing)

    return pd.concat(frames, ignore_index=True)


def get_column_context(df: pd.DataFrame, question: str, top_k: int = 12) -> str:
    """
    Build a compact schema string from the actual DataFrame.

    Selects the most relevant financial columns for the question using
    simple keyword matching, then shows the column name and up to 3
    sample values so the LLM can write accurate pandas expressions.

    Always includes 'company' and 'fiscalYear' as index columns.

    Returns a string like:
        Index columns (always available):
          company      : ['Apple', 'Microsoft']
          fiscalYear   : ['2023', '2024', '2025']

        Relevant financial columns:
          revenue      : [391035000000.0, 383285000000.0, 394328000000.0]
          netIncome    : [93736000000.0, 96995000000.0, 101956000000.0]
          ...
    """
    q = question.lower()

    # All available financial columns (exclude meta)
    financial_cols = [c for c in df.columns if c not in _META_COLS]

    # Score each column by keyword overlap with the question
    def score(col: str) -> int:
        words = col.lower()
        # Exact or partial match with question words
        return sum(1 for word in q.split() if len(word) > 3 and word in words)

    scored = sorted(financial_cols, key=score, reverse=True)
    relevant = scored[:top_k]

    # Always include the most common "total" columns if not already picked
    always_include = ["revenue", "netIncome", "totalAssets", "freeCashFlow", "operatingCashFlow"]
    for col in always_include:
        if col in financial_cols and col not in relevant:
            relevant.append(col)

    lines = ["Index columns (always available):"]
    for col in ["company", "fiscalYear"]:
        if col in df.columns:
            samples = df[col].dropna().unique().tolist()[:5]
            lines.append(f"  {col:<30} : {samples}")

    lines.append("\nRelevant financial columns:")
    for col in relevant:
        if col in df.columns:
            samples = df[col].dropna().tolist()[:3]
            lines.append(f"  {col:<30} : {samples}")

    lines.append("\nAll available financial columns (use if needed):")
    lines.append("  " + ", ".join(financial_cols))

    return "\n".join(lines)
