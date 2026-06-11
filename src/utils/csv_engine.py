"""
FinSight CSV Engine
────────────────────
Loads financial CSVs into a single combined DataFrame.
All CSVs share the same structure: a 'date' column holding the fiscal
year-end date, from which we derive a 4-digit 'fiscalYear' column.
"""

import logging
from typing import Optional

import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)

_VALID_STATEMENT_TYPES = {"income_statement", "balance_sheet", "cash_flow"}

# Null-like strings the LLM sometimes returns instead of JSON null
_NULL_STRINGS = {"null", "none", "n/a", "", "undefined"}


def sanitize_statement_type(raw) -> Optional[str]:
    """Return None if raw is null-like or not a valid statement type."""
    if raw is None:
        return None
    s = str(raw).lower().strip()
    if s in _NULL_STRINGS or s not in _VALID_STATEMENT_TYPES:
        return None
    return raw


def load_dataframes(
    companies: list[str],
    statement_type: str,
    years: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load CSVs for the given companies and statement type into a single DataFrame.

    Args:
        companies:      lowercase company names e.g. ["apple", "nvidia"]
        statement_type: "income_statement" | "balance_sheet" | "cash_flow"
        years:          optional year filter e.g. ["2023", "2024"]

    Returns:
        Combined DataFrame with a 'fiscalYear' column (4-digit string)
        and a 'company' column (title-cased).

    Raises:
        ValueError: if statement_type is invalid or no data found.
    """
    if statement_type not in _VALID_STATEMENT_TYPES:
        raise ValueError(
            f"Invalid statement_type '{statement_type}'. Must be one of: {_VALID_STATEMENT_TYPES}"
        )

    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for company in companies:
        path = cfg.CSV_PATHS.get(statement_type, {}).get(company.lower())
        if path is None or not path.exists():
            logger.warning("CSV not found — company=%s, statement=%s", company, statement_type)
            missing.append(company)
            continue

        df = pd.read_csv(path, encoding="utf-8-sig")

        # Derive fiscal year from the date column
        df["fiscalYear"] = (
            pd.to_datetime(df["date"], errors="coerce")
            .dt.year
            .astype("Int64")
            .astype(str)
            .replace("<NA>", "")
        )
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
