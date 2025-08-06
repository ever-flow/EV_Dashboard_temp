import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

from config import PERIODS
from data_utils import convert_to_usd, create_multiple_classification_data

# 2. Data Load & Pre‑processing
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=True)
def load_raw_data() -> pd.DataFrame:
    """Load Excel source & basic cleaning"""
    file_path = Path(__file__).with_name("heatmap_data_with_SE_v2.xlsx")
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    # Keep rows where at least one EMSEC column is non‑null
    emsec_cols = [f'EMSEC{i}' for i in range(1, 6)]
    df = df[df[emsec_cols].notna().any(axis=1)].copy()

    # Fallback Company column
    if 'Company' not in df.columns:
        df['Company'] = df['ticker']

    # Map market → country
    market_country_map = {
        'KOSPI': '한국', 'KOSDAQ': '한국', 'KOSDAQ GLOBAL': '한국',
        'NASDAQ': '미국', 'NYSE': '미국',
        'Prime (Domestic Stocks)': '일본', 'Standard (Domestic Stocks)': '일본',
        'Prime (Foreign Stocks)': '일본'
    }
    df['Country'] = df['market'].map(market_country_map).fillna('Unclassified')

    # Normalize market label
    df['Market'] = df['market'].replace({'KOSDAQ GLOBAL': 'KOSDAQ'})
    return df


def calculate_financial_metrics_with_currency_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Replicates the standalone v3.6 financial‑metric pipeline."""
    df = df.copy()

    def safe_divide(num, den):
        return num / den.replace(0, np.nan)

    # — absolute USD columns ---------------------------------------------------
    abs_cols = ['Market Cap (2024-12-31)', 'Enterprise Value (FQ0)']
    periods = PERIODS
    for p in periods:
        abs_cols.extend([
            f'Revenue ({p})', f'EBIT ({p})', f'Net Income ({p})',
            f'Total Assets ({p})', f'Equity ({p})', f'Total Liabilities ({p})',
            f'Net Debt ({p})', f'Depreciation ({p})', f'Dividends ({p})',
            f'Net Income After Minority ({p})'
        ])
    for col in abs_cols:
        if col in df.columns:
            df[f'{col}_USD'] = df.apply(lambda r: convert_to_usd(r[col], r['Country']), axis=1)

    # — derived metrics --------------------------------------------------------
    for p in periods:
        mc = 'Market Cap (2024-12-31)_USD'
        ebit_usd = f'EBIT ({p})_USD'
        dep_usd = f'Depreciation ({p})_USD'
        nd_usd = f'Net Debt ({p})_USD'
        ev_usd = 'Enterprise Value (FQ0)_USD'

        if ebit_usd in df.columns and dep_usd in df.columns:
            df[f'EBITDA ({p})_USD'] = df[ebit_usd] + df[dep_usd]

        if mc in df.columns and nd_usd in df.columns:
            df[ev_usd] = df.get(ev_usd, np.nan)
            mask = df[ev_usd].isna()
            df.loc[mask, ev_usd] = df.loc[mask, mc] + df.loc[mask, nd_usd].fillna(0)

        ni_usd = f'Net Income ({p})_USD'
        eq_usd = f'Equity ({p})_USD'
        ebitda_usd = f'EBITDA ({p})_USD'
        rev_usd = f'Revenue ({p})_USD'

        if mc in df.columns and ni_usd in df.columns:
            df[f'PER ({p})'] = safe_divide(df[mc], df[ni_usd])
        if mc in df.columns and eq_usd in df.columns:
            df[f'PBR ({p})'] = safe_divide(df[mc], df[eq_usd])
        if ev_usd in df.columns and ebitda_usd in df.columns:
            df[f'EV_EBITDA ({p})'] = safe_divide(df[ev_usd], df[ebitda_usd])
        if mc in df.columns and rev_usd in df.columns:
            df[f'시가총액/매출액 ({p})'] = safe_divide(df[mc], df[rev_usd])
        if mc in df.columns and ebit_usd in df.columns:
            df[f'시가총액/영업이익 ({p})'] = safe_divide(df[mc], df[ebit_usd])

        # — ratios (local currency OK) ----------------------------------------
        ni = f'Net Income ({p})'
        ni_after = f'Net Income After Minority ({p})'
        ni = ni_after if ni_after in df.columns else ni
        assets = f'Total Assets ({p})'
        liab = f'Total Liabilities ({p})'

        if ni in df.columns and eq_usd.replace('_USD', '') in df.columns:
            df[f'ROE ({p})'] = safe_divide(df[ni], df[eq_usd.replace('_USD', '')])
        if ebit_usd.replace('_USD', '') in df.columns and rev_usd.replace('_USD', '') in df.columns:
            df[f'영업이익률 ({p})'] = safe_divide(df[ebit_usd.replace('_USD', '')], df[rev_usd.replace('_USD', '')])
        if ebit_usd in df.columns and rev_usd in df.columns and dep_usd.replace('_USD', '') in df.columns:
            df[f'EBITDA/Sales ({p})'] = safe_divide(df[ebit_usd.replace('_USD', '')] + df[dep_usd.replace('_USD', '')], df[rev_usd.replace('_USD', '')])
        if ni in df.columns and assets.replace('_USD', '') in df.columns:
            df[f'총자산이익률 ({p})'] = safe_divide(df[ni], df[assets.replace('_USD', '')])
        if rev_usd.replace('_USD', '') in df.columns and assets.replace('_USD', '') in df.columns:
            df[f'자산회전율 ({p})'] = safe_divide(df[rev_usd.replace('_USD', '')], df[assets.replace('_USD', '')])
        if eq_usd.replace('_USD', '') in df.columns and assets.replace('_USD', '') in df.columns:
            df[f'자기자본비율 ({p})'] = safe_divide(df[eq_usd.replace('_USD', '')], df[assets.replace('_USD', '')])
        if liab.replace('_USD', '') in df.columns and eq_usd.replace('_USD', '') in df.columns:
            df[f'부채비율 ({p})'] = safe_divide(df[liab.replace('_USD', '')], df[eq_usd.replace('_USD', '')])

    return df


def prepare_streamlit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide → long for multi‑year access in Streamlit."""
    rows = []
    for yr in PERIODS:
        tmp = df.copy()
        tmp['Year'] = yr
        mapping = {
            f'PER ({yr})': 'PER',
            f'PBR ({yr})': 'PBR',
            f'EV_EBITDA ({yr})': 'EV_EBITDA',
            f'ROE ({yr})': 'ROE',
            f'영업이익률 ({yr})': '영업이익률',
            f'EBITDA/Sales ({yr})': 'EBITDA/Sales',
            f'총자산이익률 ({yr})': '총자산이익률',
            f'자산회전율 ({yr})': '자산회전율',
            f'자기자본비율 ({yr})': '자기자본비율',
            f'부채비율 ({yr})': '부채비율',
            f'시가총액/매출액 ({yr})': '시가총액/매출액',
            f'시가총액/영업이익 ({yr})': '시가총액/영업이익',
            f'Net Income ({yr})_USD': 'Net_Income',
            f'EBITDA ({yr})_USD': 'EBITDA',
            f'Revenue ({yr})_USD': 'Sales',
            f'Total Assets ({yr})_USD': 'Assets',
            f'Equity ({yr})_USD': 'Book'
        }
        for old, new in mapping.items():
            if old in tmp.columns:
                tmp[new] = tmp[old]
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


@st.cache_data(show_spinner=True)
def load_processed_data() -> pd.DataFrame:
    raw = load_raw_data()
    metrics = calculate_financial_metrics_with_currency_conversion(raw)

    # Identify companies with any missing financials ------------------------
    non_financial_keywords = ['EMSEC', 'EMTEC', 'ticker', 'market', 'Country', 'Market', 'name', 'Company']
    fin_cols = [c for c in metrics.columns if not any(k in c for k in non_financial_keywords)]
    company_missing = metrics.groupby('ticker')[fin_cols].apply(lambda x: x.isnull().values.any())
    metrics['has_missing_financials'] = metrics['ticker'].map(company_missing)

    expanded = create_multiple_classification_data(metrics)
    return prepare_streamlit_data(expanded)


