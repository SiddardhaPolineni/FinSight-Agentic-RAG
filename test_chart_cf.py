import sys, asyncio
sys.path.insert(0, ".")

# Test what the chart_node actually produces
from src.retriever import SchemaRetriever
from src.utils import load_dataframes
import pandas as pd
import json

sr = SchemaRetriever()
question = "plot Apple cash flow from 2023 to 2025"
result = sr.retrieve(question)
print(f"statement_type: {result['statement_type']}")
print(f"schema_context (first 300):\n{result['schema_context'][:300]}")
print()

df = load_dataframes(["apple"], result["statement_type"], ["2023", "2024", "2025"])
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"fiscalYear values: {df['fiscalYear'].unique().tolist()}")
print()

# Check if key cash flow columns have data
for col in ["freeCashFlow", "operatingCashFlow", "netChangeInCash", "capitalExpenditure"]:
    if col in df.columns:
        print(f"  {col}: {df[col].tolist()}")
    else:
        print(f"  {col}: NOT IN COLUMNS")
