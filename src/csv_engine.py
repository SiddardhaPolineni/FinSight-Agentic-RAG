"""
FinSight CSV Engine
────────────────────
Thin data-access layer. Loads the right CSV(s) as a combined DataFrame
and returns it along with the schema description from the semantic model.

The actual query logic lives in csv_node (nodes.py) — the LLM writes
a pandas expression against the DataFrame, and we execute it here.
"""

import logging
from typing import Optional

import pandas as pd

from src import config as cfg
from src.glossary import get_schema_description

logger = logging.getLogger(__name__)


def load_dataframes(
    companies: list[str],
    statement_type: str,
    years: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Load CSVs for the given companies and statement type, concatenate them
    into a single DataFrame, and return it together with the schema description.

    Args:
        companies:      list of lowercase company names e.g. ["apple", "nvidia"]
        statement_type: "income_statement" | "balance_sheet" | "cash_flow"
        years:          optional year filter e.g. ["2023", "2024"]

    Returns:
        (combined_df, schema_description_str)

    Raises:
        ValueError: if no data could be loaded for any of the companies.
    """
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for company in companies:
        path = cfg.CSV_PATHS.get(statement_type, {}).get(company.lower())
        if path is None or not path.exists():
            logger.warning("CSV not found — company=%s, statement=%s", company, statement_type)
            missing.append(company)
            continue

        df = pd.read_csv(path)
        df["fiscalYear"] = df["fiscalYear"].astype(str)
        df["company"]    = company.title()

        if years:
            df = df[df["fiscalYear"].isin(years)]

        frames.append(df)
        logger.info("Loaded %s / %s — %d rows", company, statement_type, len(df))

    if not frames:
        raise ValueError(
            f"No CSV data found for companies={companies}, "
            f"statement={statement_type}. "
            f"Missing: {missing}"
        )

    if missing:
        logger.warning("Could not load data for: %s", missing)

    combined = pd.concat(frames, ignore_index=True)

    # Schema description from semantic model (injected into the LLM prompt)
    schema_desc = get_schema_description(statement_type)

    return combined, schema_desc
