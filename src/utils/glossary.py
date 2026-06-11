"""
FinSight Financial Semantic Model
───────────────────────────────────
SEMANTIC_MODEL describes every column in the three financial statement CSVs.

Each entry:
  key              → exact CSV column name
  "statement"      → which statement file owns it
  "dtype"          → "currency" | "ratio" | "shares"
  "aliases"        → natural-language names a user might use
  "description"    → human-readable definition

get_schema_description(statement_type) returns a compact string
that is injected into the pandas-query prompt so the LLM knows
exactly which columns are available and what they mean.
"""

from __future__ import annotations

SEMANTIC_MODEL: dict[str, dict] = {

    # INCOME STATEMENT
    "revenue":                                    {"statement": "income_statement", "dtype": "currency", "aliases": ["revenue", "sales", "total revenue", "net revenue", "top line"], "description": "Total net revenue for the period."},
    "costOfRevenue":                              {"statement": "income_statement", "dtype": "currency", "aliases": ["cost of revenue", "cogs", "cost of goods sold"], "description": "Direct costs of producing goods/services."},
    "grossProfit":                                {"statement": "income_statement", "dtype": "currency", "aliases": ["gross profit", "gross income", "gross margin"], "description": "Revenue minus cost of revenue."},
    "researchAndDevelopmentExpenses":             {"statement": "income_statement", "dtype": "currency", "aliases": ["r&d", "research and development", "r&d expenses"], "description": "R&D spending."},
    "sellingGeneralAndAdministrativeExpenses":    {"statement": "income_statement", "dtype": "currency", "aliases": ["sga", "sg&a", "selling general and administrative"], "description": "SG&A expenses."},
    "operatingExpenses":                          {"statement": "income_statement", "dtype": "currency", "aliases": ["opex", "operating expenses"], "description": "Total operating expenses."},
    "operatingIncome":                            {"statement": "income_statement", "dtype": "currency", "aliases": ["operating income", "operating profit", "income from operations"], "description": "Gross profit minus operating expenses."},
    "ebit":                                       {"statement": "income_statement", "dtype": "currency", "aliases": ["ebit", "earnings before interest and tax"], "description": "Earnings before interest and taxes."},
    "ebitda":                                     {"statement": "income_statement", "dtype": "currency", "aliases": ["ebitda"], "description": "EBIT plus depreciation and amortization."},
    "interestIncome":                             {"statement": "income_statement", "dtype": "currency", "aliases": ["interest income", "interest earned"], "description": "Income from interest-bearing assets."},
    "interestExpense":                            {"statement": "income_statement", "dtype": "currency", "aliases": ["interest expense"], "description": "Interest costs on debt."},
    "depreciationAndAmortization":                {"statement": "income_statement", "dtype": "currency", "aliases": ["depreciation", "d&a", "depreciation and amortization"], "description": "Non-cash asset write-down."},
    "incomeBeforeTax":                            {"statement": "income_statement", "dtype": "currency", "aliases": ["pretax income", "income before tax", "ebt"], "description": "Earnings before income tax."},
    "incomeTaxExpense":                           {"statement": "income_statement", "dtype": "currency", "aliases": ["tax", "income tax", "tax expense"], "description": "Income tax charged against earnings."},
    "netIncome":                                  {"statement": "income_statement", "dtype": "currency", "aliases": ["net income", "net profit", "earnings", "profit", "bottom line"], "description": "After-tax profit."},
    "eps":                                        {"statement": "income_statement", "dtype": "ratio",    "aliases": ["basic eps", "basic earnings per share"], "description": "Basic EPS."},
    "epsDiluted":                                 {"statement": "income_statement", "dtype": "ratio",    "aliases": ["eps", "earnings per share", "diluted eps"], "description": "Diluted EPS."},
    "weightedAverageShsOut":                      {"statement": "income_statement", "dtype": "shares",   "aliases": ["shares outstanding", "basic shares"], "description": "Weighted average basic shares."},
    "weightedAverageShsOutDil":                   {"statement": "income_statement", "dtype": "shares",   "aliases": ["diluted shares", "diluted shares outstanding"], "description": "Weighted average diluted shares."},


    # BALANCE SHEET
    "cashAndCashEquivalents":                     {"statement": "balance_sheet", "dtype": "currency", "aliases": ["cash", "cash and equivalents", "cash and cash equivalents"], "description": "Cash and highly liquid assets."},
    "shortTermInvestments":                       {"statement": "balance_sheet", "dtype": "currency", "aliases": ["short term investments", "investments", "marketable securities"], "description": "Investments maturing within one year."},
    "cashAndShortTermInvestments":                {"statement": "balance_sheet", "dtype": "currency", "aliases": ["cash and short term investments", "liquid assets"], "description": "Cash plus short-term investments."},
    "netReceivables":                             {"statement": "balance_sheet", "dtype": "currency", "aliases": ["receivables", "accounts receivable"], "description": "Amounts owed by customers."},
    "inventory":                                  {"statement": "balance_sheet", "dtype": "currency", "aliases": ["inventory", "inventories"], "description": "Value of goods held for sale."},
    "totalCurrentAssets":                         {"statement": "balance_sheet", "dtype": "currency", "aliases": ["current assets", "total current assets"], "description": "Assets convertible to cash within one year."},
    "propertyPlantEquipmentNet":                  {"statement": "balance_sheet", "dtype": "currency", "aliases": ["ppe", "property plant and equipment", "fixed assets"], "description": "Tangible long-term assets net of depreciation."},
    "goodwill":                                   {"statement": "balance_sheet", "dtype": "currency", "aliases": ["goodwill"], "description": "Acquisition premium over fair value."},
    "intangibleAssets":                           {"statement": "balance_sheet", "dtype": "currency", "aliases": ["intangibles", "intangible assets"], "description": "Non-physical long-term assets."},
    "longTermInvestments":                        {"statement": "balance_sheet", "dtype": "currency", "aliases": ["long term investments"], "description": "Investments held over one year."},
    "totalNonCurrentAssets":                      {"statement": "balance_sheet", "dtype": "currency", "aliases": ["non current assets", "long term assets"], "description": "Assets not due within one year."},
    "totalAssets":                                {"statement": "balance_sheet", "dtype": "currency", "aliases": ["total assets", "assets"], "description": "Sum of all assets."},
    "accountPayables":                            {"statement": "balance_sheet", "dtype": "currency", "aliases": ["accounts payable", "payables"], "description": "Amounts owed to suppliers."},
    "shortTermDebt":                              {"statement": "balance_sheet", "dtype": "currency", "aliases": ["short term debt", "current debt"], "description": "Debt due within one year."},
    "totalCurrentLiabilities":                    {"statement": "balance_sheet", "dtype": "currency", "aliases": ["current liabilities", "total current liabilities"], "description": "Obligations due within one year."},
    "longTermDebt":                               {"statement": "balance_sheet", "dtype": "currency", "aliases": ["long term debt", "long-term debt"], "description": "Debt due after one year."},
    "totalNonCurrentLiabilities":                 {"statement": "balance_sheet", "dtype": "currency", "aliases": ["non current liabilities", "long term liabilities"], "description": "Obligations not due within one year."},
    "totalLiabilities":                           {"statement": "balance_sheet", "dtype": "currency", "aliases": ["total liabilities", "liabilities"], "description": "Sum of all liabilities."},
    "retainedEarnings":                           {"statement": "balance_sheet", "dtype": "currency", "aliases": ["retained earnings", "accumulated earnings"], "description": "Cumulative undistributed earnings."},
    "totalStockholdersEquity":                    {"statement": "balance_sheet", "dtype": "currency", "aliases": ["equity", "stockholders equity", "shareholders equity", "book value"], "description": "Net assets attributable to shareholders."},
    "totalEquity":                                {"statement": "balance_sheet", "dtype": "currency", "aliases": ["total equity"], "description": "Total equity including minority interest."},
    "totalDebt":                                  {"statement": "balance_sheet", "dtype": "currency", "aliases": ["total debt", "debt"], "description": "Short-term plus long-term debt."},
    "netDebt":                                    {"statement": "balance_sheet", "dtype": "currency", "aliases": ["net debt"], "description": "Total debt minus cash."},

    # CASH FLOW STATEMENT
    "stockBasedCompensation":                     {"statement": "cash_flow", "dtype": "currency", "aliases": ["stock based compensation", "sbc", "share based compensation"], "description": "Non-cash equity compensation expense."},
    "changeInWorkingCapital":                     {"statement": "cash_flow", "dtype": "currency", "aliases": ["change in working capital", "working capital change"], "description": "Net change in current assets minus liabilities."},
    "netCashProvidedByOperatingActivities":       {"statement": "cash_flow", "dtype": "currency", "aliases": ["cash from operations", "operating activities"], "description": "Cash from core business operations."},
    "operatingCashFlow":                          {"statement": "cash_flow", "dtype": "currency", "aliases": ["operating cash flow", "cfo", "cash flow from operations"], "description": "Cash generated from operations."},
    "capitalExpenditure":                         {"statement": "cash_flow", "dtype": "currency", "aliases": ["capex", "capital expenditure", "capital expenditures", "capital spending"], "description": "Cash spent on long-term physical assets."},
    "investmentsInPropertyPlantAndEquipment":     {"statement": "cash_flow", "dtype": "currency", "aliases": ["ppe investment"], "description": "Cash spent on PP&E."},
    "netCashProvidedByInvestingActivities":       {"statement": "cash_flow", "dtype": "currency", "aliases": ["investing activities", "cash from investing"], "description": "Net cash from investing activities."},
    "commonStockRepurchased":                     {"statement": "cash_flow", "dtype": "currency", "aliases": ["buybacks", "stock buybacks", "share repurchases"], "description": "Cash spent repurchasing shares."},
    "netDividendsPaid":                           {"statement": "cash_flow", "dtype": "currency", "aliases": ["dividends paid", "dividends"], "description": "Cash paid as dividends."},
    "netCashProvidedByFinancingActivities":       {"statement": "cash_flow", "dtype": "currency", "aliases": ["financing activities", "cash from financing"], "description": "Net cash from financing activities."},
    "netChangeInCash":                            {"statement": "cash_flow", "dtype": "currency", "aliases": ["net change in cash", "change in cash"], "description": "Net increase/decrease in cash."},
    "freeCashFlow":                               {"statement": "cash_flow", "dtype": "currency", "aliases": ["free cash flow", "fcf"], "description": "Operating cash flow minus capex."},
    "incomeTaxesPaid":                            {"statement": "cash_flow", "dtype": "currency", "aliases": ["taxes paid", "income taxes paid"], "description": "Actual cash paid for income taxes."},
}


def get_schema_description(statement_type: str) -> str:
    """
    Return a compact schema string for the given statement type,
    ready to inject into an LLM prompt.

    Format per line:
      column_name (dtype) — aliases: [...] — description

    Example output for income_statement:
      revenue (currency) — aliases: [revenue, sales, top line] — Total net revenue for the period.
      netIncome (currency) — aliases: [net income, profit, bottom line] — After-tax profit.
      ...
    """
    lines = []
    for col, meta in SEMANTIC_MODEL.items():
        if meta["statement"] == statement_type:
            alias_str = ", ".join(meta["aliases"][:4])   # cap at 4 to keep prompt compact
            lines.append(
                f"  {col} ({meta['dtype']}) — "
                f"aliases: [{alias_str}] — "
                f"{meta['description']}"
            )
    return "\n".join(lines)
