import os, pathlib, tempfile

# 1) HOME이 없거나 '/'면 강제로 교체
if os.environ.get("HOME", "/") in ("", "/"):
    os.environ["HOME"] = "/tmp"             # 필요하면 tempfile.mkdtemp() 사용

# 2) .streamlit 디렉터리 생성
streamlit_dir = pathlib.Path(os.environ["HOME"], ".streamlit")
streamlit_dir.mkdir(parents=True, exist_ok=True)

# 3) 사용 통계 비활성화
os.environ["STREAMLIT_DISABLE_USAGE_STATS"] = "true"


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ast
from typing import List
from pathlib import Path
import re

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Integrated EMSEC × EMTEC Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────────────────────────
# 0. Constants
# ─────────────────────────────────────────────────────────────────────────────
PERIODS = ["LTM", "LTM-1", "LTM-2", "LTM-3"]
METHODS = ["AVG", "MED", "HRM", "AGG"]
COLORS = {"AVG": "#1f77b4", "MED": "#ff7f0e", "HRM": "#2ca02c", "AGG": "#d62728"}
EXCHANGE_RATES = {  # 2024‑12‑31 기준 환율 (원/달러, 엔/달러)
    '한국': 1380.0,
    '미국': 1.0,
    '일본': 157.0,
    'Unclassified': 1.0
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_usd(value, country):
    """Convert local currency to USD using EXCHANGE_RATES"""
    if pd.isna(value) or pd.isna(country):
        return value
    return value / EXCHANGE_RATES.get(country, 1.0)


def parse_emtec_list(txt: str) -> List[str]:
    """Safely parse the EMTEC list string"""
    try:
        return [] if pd.isna(txt) or txt in ('[]', '') else ast.literal_eval(txt)
    except Exception:
        return []



def create_multiple_classification_data(df: pd.DataFrame) -> pd.DataFrame:
    """Generate one row per every EMSEC × EMTEC combination (cross‑product)."""
    expanded_rows: List[pd.Series] = []

    for _, row in df.iterrows():
        # ── 1) Collect valid EMSEC levels ------------------------------------------------
        emsec_list = []
        for i in range(1, 6):
            emsec_code = row.get(f'EMSEC{i}')
            if pd.notna(emsec_code):
                emsec_list.append({
                    'sector': row.get(f'EMSEC{i}_Sector', 'Unclassified') or 'Unclassified',
                    'industry': row.get(f'EMSEC{i}_Industry', 'Unclassified') or 'Unclassified',
                    'sub_industry': emsec_code
                })

        # ── 2) Collect EMTEC hierarchy ---------------------------------------------------
        lvl1 = parse_emtec_list(row.get('EMTEC_LEVEL1'))
        lvl2 = parse_emtec_list(row.get('EMTEC_LEVEL2'))
        lvl3 = parse_emtec_list(row.get('EMTEC_LEVEL3'))

        emtec_combos: List[dict] = []
        if lvl1:
            for l1 in lvl1:
                for l2 in (lvl2 or ['Unclassified']):
                    for l3 in (lvl3 or ['Unclassified']):
                        emtec_combos.append({'theme': l1, 'technology': l2, 'sub_technology': l3})
        else:
            emtec_combos.append({'theme': 'Unclassified', 'technology': 'Unclassified', 'sub_technology': 'Unclassified'})

        # ── 3) Produce cross‑product rows ----------------------------------------------
        for emsec in emsec_list:
            for emtec in emtec_combos:
                new_row = row.copy()
                new_row['Sector'] = emsec['sector']
                new_row['Industry'] = emsec['industry']
                new_row['Sub_industry'] = emsec['sub_industry']
                new_row['Theme'] = emtec['theme']
                new_row['Technology'] = emtec['technology']
                new_row['Sub_Technology'] = emtec['sub_technology']
                expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows)


# ─────────────────────────────────────────────────────────────────────────────
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


DF_RAW = load_processed_data()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Aggregation Logic (unchanged from v3.6)
# ─────────────────────────────────────────────────────────────────────────────

def compute_aggregate(sub_df, metric, agg_func, year_sel, group_sel, metric_main, metric_mode, base_col):
    """Exact logic from standalone Heatmap (v3.6)."""
    if group_sel == "기업":
        if metric_main == "기업수":
            total = sub_df['Company'].nunique()
            if total == 0:
                return 0 if metric_mode != "결측 비율" else np.nan
            if metric_mode == "결측 포함":
                return total
            elif metric_mode == "결측 미포함":
                return sub_df.loc[~sub_df['has_missing_financials'], 'Company'].nunique()
            elif metric_mode == "결측 비율":
                missing = sub_df.loc[sub_df['has_missing_financials'], 'Company'].nunique()
                return missing / total if total else np.nan
        elif metric_main == "0이하비율":
            arr = pd.to_numeric(sub_df[base_col], errors="coerce")
            if metric_mode == "결측 제외":
                arr = arr.dropna()
            return (arr <= 0).sum() / len(arr) if len(arr) else np.nan

    else:  # 멀티플·재무비율
        if agg_func == "AGG":
            mc_col = 'Market Cap (2024-12-31)_USD'
            if mc_col not in sub_df.columns or sub_df[mc_col].isna().all():
                return np.nan
            mc = sub_df[mc_col]
            q1, q3 = mc.quantile(0.25), mc.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 2*iqr, q3 + 2*iqr
            filt = sub_df[(mc >= lower) & (mc <= upper)]
            if filt.empty:
                return np.nan
            if metric in ['PER', 'PBR', 'EV_EBITDA']:
                if metric == 'PER':
                    num, den = filt[mc_col].sum(), filt['Net_Income'].sum()
                elif metric == 'PBR':
                    num, den = filt[mc_col].sum(), filt['Book'].sum()
                else:  # EV_EBITDA
                    num, den = filt['Enterprise Value (FQ0)_USD'].sum(), filt['EBITDA'].sum()
                return num / den if den else np.nan
            arr = filt[metric].dropna()
            return arr.sum() if len(arr) else np.nan
        else:  # AVG / MED / HRM
            arr = sub_df[metric].dropna()
            if not len(arr):
                return np.nan
            if agg_func == 'AVG':
                return arr.mean()
            elif agg_func == 'MED':
                return arr.median()
            else:  # HRM
                arr = arr[arr > 0]
                return len(arr) / (1/arr).sum() if len(arr) else np.nan

# ---------------------------------------------------------------------------
# 4. UI Routing  (Heatmap / Scatter / 성장 패턴 / 규모 변수)
# ---------------------------------------------------------------------------
# NOTE: 시각화 2~4 로직은 원본 통합코드를 그대로 유지합니다.  Heatmap 부분만
#       create_multiple_classification_data 교정으로 정상화됩니다.
# ---------------------------------------------------------------------------

# ◀ 사이드바 공통 (Visualization choice) ▶
st.sidebar.title("시각화 선택")
visualization = st.sidebar.radio("선택하세요:", ["히트맵", "점도표", "성장 패턴", "규모 변수"])

# ================================= Heatmap =================================
if visualization == "히트맵":
    st.header("EMSEC × EMTEC Heatmap")
    # —--- Sidebar Controls (unchanged) --------------------------------------
    with st.sidebar:
        st.markdown("**기준 연도**")
        year_sel = st.selectbox("", PERIODS, label_visibility="collapsed", key="year_sel1")

        st.markdown("**상장시장**")
        market_options = [
            "전체", "한국 전체", "KOSPI", "KOSDAQ", "미국 전체", "NASDAQ",
            "일본 전체", "Prime (Domestic Stocks)", "Standard (Domestic Stocks)", "Prime (Foreign Stocks)",
        ]
        market_sel = st.selectbox("", market_options, label_visibility="collapsed", key="market_sel1")
        country_filter = market_filter = None
        if "한국" in market_sel:
            country_filter = "한국"
        elif "미국" in market_sel:
            country_filter = "미국"
        elif "일본" in market_sel:
            country_filter = "일본"
        if market_sel not in ["전체", "한국 전체", "미국 전체", "일본 전체"]:
            market_filter = market_sel

        # Row (EMSEC)
        st.markdown("**Sector > Industry**")
        available_sectors = sorted([s for s in DF_RAW.Sector.unique() if pd.notna(s) and s != 'Unclassified'])
        sector_sel = st.selectbox("", ["전체"] + available_sectors, label_visibility="collapsed", key="sector_sel1")
        if sector_sel == "전체":
            industry_pool = sorted([i for i in DF_RAW.Industry.unique() if pd.notna(i) and i != 'Unclassified'])
        else:
            industry_pool = sorted([i for i in DF_RAW.loc[DF_RAW.Sector == sector_sel, "Industry"].unique() if pd.notna(i) and i != 'Unclassified'])
        industry_sel = st.selectbox("", ["전체"] + industry_pool, label_visibility="collapsed", key="industry_sel1")

        # Column (EMTEC)
        st.markdown("**Theme > Technology**")
        available_themes = sorted([t for t in DF_RAW.Theme.unique() if pd.notna(t) and t != 'Unclassified'])
        theme_sel = st.selectbox("", ["전체"] + available_themes, label_visibility="collapsed", key="theme_sel1")
        if theme_sel == "전체":
            tech_pool = sorted([t for t in DF_RAW.Technology.unique() if pd.notna(t) and t != 'Unclassified'])
        else:
            tech_pool = sorted([t for t in DF_RAW.loc[DF_RAW.Theme == theme_sel, "Technology"].unique() if pd.notna(t) and t != 'Unclassified'])
        tech_sel = st.selectbox("", ["전체"] + tech_pool, label_visibility="collapsed", key="tech_sel1")

        st.markdown("**계측값 선택**")
        group_sel = st.selectbox("", ["기업", "비교가치 멀티플", "재무비율"], label_visibility="collapsed", key="group_sel1")
        metric_main = metric_mode = base_col = agg_func = None
        allow_subtotal = True
        if group_sel == "기업":
            corp_first = st.selectbox("", ["기업수", "0이하 비율"], label_visibility="collapsed", key="corp_first1")
            if corp_first == "기업수":
                metric_main = "기업수"
                metric_mode = st.selectbox("", ["결측 포함", "결측 미포함", "결측 비율"], label_visibility="collapsed", key="metric_mode1")
            else:
                metric_main = "0이하비율"
                base_map = {"순이익": "Net_Income", "EBITDA": "EBITDA", "매출": "Sales", "자산총계": "Assets", "순자산": "Book"}
                base_sel = st.selectbox("", list(base_map.keys()), label_visibility="collapsed", key="corp_base1")
                base_col = base_map[base_sel]
                metric_mode = st.selectbox("", ["결측 포함", "결측 제외"], label_visibility="collapsed", key="metric_mode2")
        elif group_sel == "비교가치 멀티플":
            metric_main = st.selectbox("", ["PER", "PBR", "EV_EBITDA"], label_visibility="collapsed", key="metric_main1")
            agg_func = st.selectbox("", ["AVG", "HRM", "MED", "AGG"], label_visibility="collapsed", key="agg_func1")
            allow_subtotal = agg_func == "AGG"
        else:
            metric_main = st.selectbox("", ["ROE", "영업이익률", "EBITDA/Sales", "총자산이익률", "자산회전율", "자기자본비율", "부채비율", "시가총액/매출액", "시가총액/영업이익"], label_visibility="collapsed", key="metric_main2")
            agg_func = st.selectbox("", ["AVG", "HRM", "MED", "AGG"], label_visibility="collapsed", key="agg_func2")
            allow_subtotal = agg_func == "AGG"

    # —--- DataFrame Filtering ------------------------------------------------
    DF = DF_RAW[DF_RAW.Year == year_sel].copy()
    non_financial_keywords = ['EMSEC', 'EMTEC', 'ticker', 'market', 'Country', 'Market', 'name', 'Company']
    fin_cols = [c for c in DF.columns if not any(k in c for k in non_financial_keywords)]
    company_missing = DF.groupby('ticker')[fin_cols].apply(lambda x: x.isnull().values.any())
    DF['has_missing_financials'] = DF['ticker'].map(company_missing)

    if country_filter: DF = DF[DF.Country == country_filter]
    if market_filter: DF = DF[DF.Market == market_filter]
    if sector_sel != "전체": DF = DF[DF.Sector == sector_sel]
    if industry_sel != "전체": DF = DF[DF.Industry == industry_sel]
    if theme_sel != "전체": DF = DF[DF.Theme == theme_sel]
    if tech_sel != "전체": DF = DF[DF.Technology == tech_sel]

    # Index levels -----------------------------------------------------------
    row_index = "Sector" if sector_sel == "전체" else "Industry" if industry_sel == "전체" else "Sub_industry"
    col_index = "Theme" if theme_sel == "전체" else "Technology" if tech_sel == "전체" else "Sub_Technology"

    values_col = (base_col if group_sel == "기업" and metric_main == "0이하비율" else metric_main if group_sel != "기업" else "Company")
    if group_sel != "기업" and values_col not in DF.columns:
        st.warning(f"'{values_col}' 지표를 계산할 수 없습니다. 데이터나 설정을 확인해주세요.")
        st.stop()
    if DF.empty:
        st.warning("조건에 맞는 데이터가 없습니다.")
        st.stop()

    # Pivot (main values & counts) -------------------------------------------
    pivot_main = DF.groupby([row_index, col_index]).apply(
        lambda g: compute_aggregate(g, values_col, agg_func, year_sel, group_sel, metric_main, metric_mode, base_col)
    ).unstack(fill_value=np.nan)
    pivot_counts = DF.groupby([row_index, col_index])['Company'].nunique().unstack(fill_value=0)

    if pivot_main.empty:
        st.warning("피벗 테이블을 생성할 수 없습니다.")
        st.stop()

    x_orig, y_orig = pivot_main.columns.tolist(), pivot_main.index.tolist()
    z_core = pivot_main.values
    cnt_core = pivot_counts.reindex(index=y_orig, columns=x_orig).fillna(0).values

    # Subtotals/GrandTotals ---------------------------------------------------
    if allow_subtotal:
        row_tot = DF.groupby(row_index).apply(lambda g: compute_aggregate(g, values_col, agg_func, year_sel, group_sel, metric_main, metric_mode, base_col)).reindex(y_orig)
        col_tot = DF.groupby(col_index).apply(lambda g: compute_aggregate(g, values_col, agg_func, year_sel, group_sel, metric_main, metric_mode, base_col)).reindex(x_orig)
        grand_tot = compute_aggregate(DF, values_col, agg_func, year_sel, group_sel, metric_main, metric_mode, base_col)
        row_cnt = DF.groupby(row_index)['Company'].nunique().reindex(y_orig)
        col_cnt = DF.groupby(col_index)['Company'].nunique().reindex(x_orig)
        grand_cnt = DF['Company'].nunique()

        x_labels = ["Subtotal"] + x_orig
        y_labels = ["Subtotal"] + y_orig
        sz = (len(y_labels), len(x_labels))
        z_main, cnt_main = np.full(sz, np.nan), np.full(sz, 0)
        z_main[1:, 1:], cnt_main[1:, 1:] = z_core, cnt_core
        z_sub, cnt_sub = np.full(sz, np.nan), np.full(sz, 0)
        z_sub[0, 1:], z_sub[1:, 0] = col_tot.values, row_tot.values
        cnt_sub[0, 1:], cnt_sub[1:, 0] = col_cnt.values, row_cnt.values
        z_grd, cnt_grd = np.full(sz, np.nan), np.full(sz, 0)
        z_grd[0, 0], cnt_grd[0, 0] = grand_tot, grand_cnt
        z_comb = np.where(np.isnan(z_main), z_sub, z_main)
        z_comb[0, 0] = grand_tot
    else:
        x_labels, y_labels = x_orig, y_orig
        z_main, cnt_main = z_core, cnt_core
        z_comb, z_sub, z_grd, cnt_sub, cnt_grd = z_main, None, None, None, None

    # Value formatter ---------------------------------------------------------
    if group_sel == "기업":
        if metric_main == "기업수":
            fmt = (lambda v: f"{v:,}" if pd.notna(v) else "") if metric_mode in ["결측 포함", "결측 미포함"] else (lambda v: f"{v*100:.1f}%" if pd.notna(v) else "")
        else:
            fmt = lambda v: f"{v*100:.1f}%" if pd.notna(v) else ""
    elif metric_main in ["PER", "PBR", "EV_EBITDA", "시가총액/매출액", "시가총액/영업이익", "자산회전율"]:
        fmt = lambda v: f"{v:,.2f}x" if pd.notna(v) else ""
    elif metric_main in ["ROE", "영업이익률", "EBITDA/Sales", "총자산이익률", "자기자본비율", "부채비율"] and agg_func != 'AGG':
        fmt = lambda v: f"{v*100:.1f}%" if pd.notna(v) else ""
    else:
        fmt = lambda v: f"${v:,.0f}" if pd.notna(v) else ""

    txt = [[fmt(v) for v in row] for row in z_comb]

    # Plotly Heatmap ----------------------------------------------------------
    MAIN_CS = ["#e3f2fd", "#bbdefb", "#90caf9", "#64b5f6", "#42a5f5", "#2196f3", "#1e88e5", "#1976d2", "#1565c0", "#0d47a1"]
    SUB_CS = ["#f0f0f0", "#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252"]
    GT_CS = [[0, "#000000"], [1, "#000000"]]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=z_main, x=x_labels, y=y_labels, colorscale=MAIN_CS,
                             colorbar=dict(title=metric_main), customdata=cnt_main,
                             hovertemplate="%{y} / %{x}<br>값: %{z:,.3f}<br>기업수: %{customdata:,}", xgap=1, ygap=1, hoverongaps=False))
    if allow_subtotal:
        fig.add_trace(go.Heatmap(z=z_sub, x=x_labels, y=y_labels, colorscale=SUB_CS,
                                 showscale=False, customdata=cnt_sub,
                                 hovertemplate="%{y} / %{x}<br>Subtotal: %{z:,.3f}<br>기업수: %{customdata:,}", xgap=1, ygap=1, hoverongaps=False))
        fig.add_trace(go.Heatmap(z=z_grd, x=x_labels, y=y_labels, colorscale=GT_CS,
                                 showscale=False, customdata=cnt_grd,
                                 hovertemplate="Grand Total<br>값: %{z:,.3f}<br>기업수: %{customdata:,}", xgap=1, ygap=1, hoverongaps=False))

    # Annotations ------------------------------------------------------------
    annotations = []
    for r, row in enumerate(z_comb):
        for c, val in enumerate(row):
            if pd.isna(val):
                continue
            is_grand = allow_subtotal and r == 0 and c == 0
            color = "white" if is_grand else "black"
            annotations.append(dict(text=txt[r][c], x=x_labels[c], y=y_labels[r], showarrow=False, font=dict(color=color)))

    fig.update_layout(annotations=annotations,
                      height=max(650, 35*len(y_labels)),
                      margin=dict(l=40, r=40, t=40, b=40),
                      xaxis=dict(side="top", showgrid=False),
                      yaxis=dict(autorange="reversed", showgrid=False, categoryorder="array", categoryarray=y_labels),
                      showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    unique_companies = DF['Company'].nunique()
    total_combos = len(DF)
    st.caption(f"고유 기업 수: {unique_companies:,} | 총 분류 조합 수: {total_combos:,} | 절대값 지표: USD 기준")



# --- 시각화 2: Scatter Plot ---
elif visualization == "점도표":
    st.header("Scatter Plot")
    
    METRIC_HIERARCHY = {
        "Valuation": ["PER", "PBR", "EV_EBITDA", "시가총액/매출액", "시가총액/영업이익"],
        "Profitability": ["ROE", "영업이익률", "EBITDA/Sales", "총자산이익률"],
        "Activity": ["자산회전율"],
        "Stability": ["자기자본비율", "부채비율"]
    }

    with st.sidebar:
        st.markdown("### 분류")
        classification_type = st.radio("분석 기준", ["EMSEC", "EMTEC"], horizontal=True, key="class_type2")
        cls_df = DF_RAW.copy()  # Classification 열을 직접 사용하지 않음
        if classification_type == "EMSEC":
            l1_options = ["전체"] + sorted(cls_df["Sector"].dropna().unique())
            l1_label = "Sector"
            l1_selection = st.selectbox(l1_label, l1_options, key="sector_sel2")
            l2_label = "Industry"
        else:
            l1_options = ["전체"] + sorted(cls_df["Theme"].dropna().unique())
            l1_label = "Theme"
            l1_selection = st.selectbox(l1_label, l1_options, key="theme_sel2")
            l2_label = "Technology"

        st.markdown("### 설정")
        period_sel = st.selectbox("기간", PERIODS, key="period_sel2")
        metric_group = st.selectbox("지표 그룹", list(METRIC_HIERARCHY.keys()), key="metric_group2")
        metric_options = METRIC_HIERARCHY[metric_group]
        metric_sel = st.selectbox("지표", metric_options, key="metric_sel2")
        country_sel = st.selectbox("국가", ["전체", "한국", "미국", "일본"], key="country_sel2")
        market_pool_options = {
            "전체": ["전체"] + sorted(cls_df["Market"].dropna().unique()),
            "한국": ["전체", "KOSPI", "KOSDAQ"],
            "미국": ["전체", "NASDAQ", "NYSE"],
            "일본": ["전체", "Prime (Domestic Stocks)", "Standard (Domestic Stocks)", "Prime (Foreign Stocks)"],
        }
        market_sel = st.selectbox("거래소", market_pool_options.get(country_sel, ["전체"]), key="market_sel2")

    def filter_data(df: pd.DataFrame) -> pd.DataFrame:
        d = df[df.Year == period_sel].copy()
        if classification_type == "EMSEC":
            if l1_selection != "전체":
                d = d[d["Sector"] == l1_selection]
        else:
            if l1_selection != "전체":
                d = d[d["Theme"] == l1_selection]
        if country_sel != "전체":
            d = d[d["Country"] == country_sel]
        if market_sel != "전체":
            d = d[d["Market"] == market_sel]
        return d

    FILT_DATA = filter_data(DF_RAW)
    metric_col = metric_sel
    if metric_col not in FILT_DATA.columns:
        st.error(f"'{metric_col}' 열이 데이터에 없습니다. 데이터나 설정을 확인해주세요.")
        st.stop()

    def harmonic_mean(arr: pd.Series):
        arr = arr.dropna()
        arr = arr[arr > 0]
        return len(arr) / (1 / arr).sum() if len(arr) > 0 else np.nan

    def aggregate_by_group(sub: pd.DataFrame, metric_col: str) -> pd.Series:
        sub_unique = sub.drop_duplicates(subset=["ticker"])
        arr = pd.to_numeric(sub_unique[metric_col], errors="coerce")
        res = {
            "AVG": arr.mean(),
            "MED": arr.median(),
            "HRM": harmonic_mean(arr)
        }
        num, den = None, None
        if metric_sel == "PER":
            num = sub_unique["Market Cap (2024-12-31)_USD"].sum()
            den = sub_unique["Net_Income"].sum()
        elif metric_sel == "PBR":
            num = sub_unique["Market Cap (2024-12-31)_USD"].sum()
            den = sub_unique["Book"].sum()
        elif metric_sel == "EV_EBITDA":
            num = sub_unique["Enterprise Value (FQ0)_USD"].sum()
            den = sub_unique["EBITDA"].sum()
        if num is not None and den is not None and den != 0:
            res["AGG"] = num / den
        else:
            res["AGG"] = res["AVG"]
        res["기업 수"] = len(arr.dropna())
        return pd.Series(res)

    if FILT_DATA.empty:
        st.warning("선택하신 조건에 맞는 데이터가 없습니다.")
        st.stop()

    agg_df = FILT_DATA.groupby(l2_label).apply(lambda g: aggregate_by_group(g, metric_col))
    agg_df = agg_df.dropna(how='all', subset=METHODS).sort_index()

    st.caption(f"분석 기준: {classification_type} > {l1_selection}")
    if agg_df.empty:
        st.warning("집계 결과 데이터가 없어 차트를 그릴 수 없습니다.")
        st.stop()

    fig = go.Figure()
    for method in METHODS:
        fig.add_trace(go.Scatter(
            x=agg_df.index,
            y=agg_df[method],
            customdata=agg_df[['기업 수']].to_numpy(),
            mode="markers",
            marker=dict(size=12, color=COLORS[method], line=dict(width=1, color="white")),
            name=method,
            hovertemplate=f"<b>{agg_df.index.name}:</b> %{{x}}<br><b>{metric_sel}:</b> %{{y:.2f}}<br><b>계산 방식:</b> {method}<br><b>기업 수:</b> %{{customdata[0]}}<br><extra></extra>"
        ))
    y_axis_title = f"{metric_sel} ({period_sel})"
    fig.update_layout(
        title=dict(text=f"{l2_label}별 '{y_axis_title}' 비교 (계산 방식별)", x=0.5, xanchor='center'),
        xaxis_title=l2_label,
        yaxis_title=y_axis_title,
        xaxis_tickangle=-45,
        legend_title="계산 방식",
        height=650,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    sel_list = [v for v in [classification_type, l1_selection, period_sel, metric_sel, country_sel if country_sel != "전체" else None, market_sel if market_sel != "전체" else None] if v and v != "전체"]
    st.caption(" | ".join(sel_list) + f" • 그룹 수: {len(agg_df):,}")

# --- 시각화 3: 성장 패턴 ---
elif visualization == "성장 패턴":
    st.header("성장 패턴 분석")
    
    def growth_symbol(x):
        return "▲" if x >= 5 else "▼" if x <= -5 else "▬"
    def make_pattern(r):
        return "".join(growth_symbol(r[f"Growth_LTM-{i}"]) for i in (3,2,1))
    pattern_category_map = {
        "▲▲▲":"Good","▬▲▲":"Good","▲▬▲":"Up","▬▬▲":"Up","▼▲▲":"Turn Up",
        "▼▬▲":"Recent Up","▬▼▲":"Recent Up","▼▼▲":"Recent Up","▲▼▲":"Recent Up",
        "▲▲▬":"Up & Flat","▬▲▬":"Up & Flat","▼▲▬":"Up & Flat","▲▬▬":"Up & Flat",
        "▬▬▬":"Flat",
        "▼▬▬":"Down & Flat","▲▼▬":"Down & Flat","▬▼▬":"Down & Flat","▼▼▬":"Down & Flat",
        "▼▲▼":"Recent Down","▲▲▼":"Recent Down","▬▲▼":"Recent Down","▲▬▼":"Recent Down",
        "▲▼▼":"Turn Down","▬▬▼":"Down","▼▬▼":"Down","▬▼▼":"Bad","▼▼▼":"Bad"
    }
    category_order = ["Good","Up","Turn Up","Recent Up","Up & Flat","Flat","Down & Flat","Recent Down","Turn Down","Down","Bad","기타"]
    growth_weights = {"Good":3,"Up":2,"Turn Up":1.5,"Recent Up":1,"Up & Flat":0.5,"Flat":0,"Down & Flat":-0.5,"Recent Down":-1,"Turn Down":-1.5,"Down":-2,"Bad":-3,"기타":0}
    color_map = dict(zip(category_order, [(255,0,0),(255,64,48),(255,128,64),(255,192,128),(192,224,192),(128,128,128),(192,160,224),(224,160,192),(224,128,128),(192,64,64),(0,0,255),(192,192,192)]))

    with st.sidebar:
        market_sel = st.selectbox("상장시장", ["한국 전체","KOSPI","KOSDAQ","미국 전체","NASDAQ","일본 전체","Prime (Domestic Stocks)","Standard (Domestic Stocks)","Prime (Foreign Stocks)"], key="market_sel3")
        if market_sel in ["한국 전체","KOSPI","KOSDAQ"]:
            country_filter = "한국"
        elif market_sel in ["미국 전체","NASDAQ"]:
            country_filter = "미국"
        elif market_sel in ["일본 전체","Prime (Domestic Stocks)","Standard (Domestic Stocks)","Prime (Foreign Stocks)"]:
            country_filter = "일본"
        else:
            country_filter = "Unclassified"
        market_filter = None if market_sel.endswith("전체") else market_sel
        class_type = st.radio("분류 체계", ["EMSEC","EMTEC"], horizontal=True, key="class_type3")
        if class_type == "EMSEC":
            sectors = sorted(DF_RAW["Sector"].dropna().unique())
            sector_sel = st.selectbox("Sector", ["전체"] + sectors, key="sector_sel3")
            if sector_sel != "전체":
                industries = sorted(DF_RAW.loc[DF_RAW["Sector"]==sector_sel, "Industry"].dropna().unique())
                industry_sel = st.selectbox("Industry", ["전체"] + industries, key="industry_sel3")
            else:
                industry_sel = "전체"
            row_level = "Sector" if sector_sel=="전체" else "Industry"
        else:
            themes = sorted(DF_RAW["Theme"].dropna().unique())
            theme_sel = st.selectbox("Theme", ["전체"] + themes, key="theme_sel3")
            if theme_sel != "전체":
                techs = sorted(DF_RAW.loc[DF_RAW["Theme"]==theme_sel, "Technology"].dropna().unique())
                tech_sel = st.selectbox("Technology", ["전체"] + techs, key="tech_sel3")
            else:
                tech_sel = "전체"
            row_level = "Theme" if theme_sel=="전체" else "Technology"
        metric_sel = st.selectbox("지표", ["Net Income","EBITDA","EBIT","Revenue"], key="metric_sel3")

    def get_cols(df: pd.DataFrame, base: str) -> List[str]:
        pat = re.compile(re.sub(r"\s+","",base) + r"\s*\(LTM(?:-\d)?\)_USD", re.I)
        cols = [c for c in df.columns if pat.search(re.sub(r"\s+","",c))]
        if len(cols) < 4:
            expected_cols = [f"{base} (LTM-{i})_USD" for i in range(0, 4)]
            st.warning(f"{base} 기준 LTM~LTM-3 열이 충분하지 않습니다: {cols}. 기본값으로 대체합니다.")
            return expected_cols[:4]
        def keyfn(c):
            m = re.search(r"LTM-(\d)", c)
            return int(m.group(1)) if m else -1
        return sorted(cols, key=keyfn)[:4]

    def calc_growth(sub: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        g = pd.DataFrame({"ticker": sub["ticker"]})
        for i in (1,2,3):
            if cols[i-1] in sub.columns and cols[i] in sub.columns:
                g[f"Growth_LTM-{i}"] = (sub[cols[i-1]] - sub[cols[i]]) / sub[cols[i]].replace(0, np.nan) * 100
            else:
                g[f"Growth_LTM-{i}"] = np.nan
        return g.replace([np.inf,-np.inf], np.nan).dropna()

    def build_growth(metric: str, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if metric == "EBITDA":
            ecols = get_cols(d, "EBIT")
            dcols = get_cols(d, "Depreciation")
            for i, (e, dcol) in enumerate(zip(ecols, dcols)):
                d[f"EBITDA_{i}"] = d.get(e, 0) + d.get(dcol, 0)
            return calc_growth(d, [f"EBITDA_{i}" for i in range(4)])
        base_map = {"Net Income": "Net Income", "EBIT": "EBIT", "Revenue": "Revenue"}
        cols = get_cols(d, base_map[metric])
        return calc_growth(d, cols)

    DF = DF_RAW[(DF_RAW["Country"] == country_filter) & ((market_filter is None) | (DF_RAW["Market"] == market_filter))].copy()
    if class_type == "EMSEC":
        if sector_sel != "전체": DF = DF[DF["Sector"] == sector_sel]
        if industry_sel != "전체": DF = DF[DF["Industry"] == industry_sel]
    else:
        if theme_sel != "전체": DF = DF[DF["Theme"] == theme_sel]
        if tech_sel != "전체": DF = DF[DF["Technology"] == tech_sel]
    if DF.empty:
        st.warning("조건에 맞는 데이터가 없습니다.")
        st.stop()

    growth_df = build_growth(metric_sel, DF)
    rows_df = DF[[row_level,"ticker"]].drop_duplicates()
    merged = rows_df.merge(growth_df, on="ticker").dropna()
    if merged.empty:
        st.warning("해당 조건·지표에서 성장률 데이터가 없습니다.")
        st.stop()

    merged["패턴"] = merged.apply(make_pattern, axis=1)
    merged["카테고리"] = merged["패턴"].map(pattern_category_map).fillna("기타")
    ratio = (merged.groupby([row_level,"카테고리"])["ticker"].nunique().reset_index(name="count").pipe(lambda x: x.merge(merged.groupby(row_level)["ticker"].nunique().reset_index(name="total"), on=row_level)))
    ratio["rate"] = ratio["count"] / ratio["total"]
    pv = ratio.pivot(index=row_level, columns="카테고리", values="rate").fillna(0)
    pv = pv[[c for c in category_order if c in pv.columns]]
    scores = pv.mul([growth_weights[c] for c in pv.columns], axis=1).sum(axis=1)
    pv = pv.loc[scores.sort_values(ascending=False).index]

    bars = [go.Bar(y=pv.index, x=pv[cat], name=cat, orientation="h", marker_color=f"rgb{color_map[cat]}") for cat in pv.columns]
    fig = go.Figure(bars).update_layout(
        barmode="stack",
        height=max(600, 40*len(pv)),
        title=f"{metric_sel} 3년 성장 패턴 ({row_level})",
        xaxis_title="비율",
        yaxis_title=row_level,
        legend_title="카테고리",
        xaxis=dict(tickformat=".0%")
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"기업 수: {merged['ticker'].nunique():,} | 행 레벨: {row_level}")
    with st.expander("📋 비율 테이블"):
        st.dataframe(pv.style.format("{:.1%}"), use_container_width=True)

# --- 시각화 4: 규모 변수 ---
elif visualization == "규모 변수":
    st.header("규모 변수 Heatmap")
    
    with st.sidebar:
        year_sel = st.selectbox("기준 연도", ["LTM","LTM-1","LTM-2","LTM-3"], key="year_sel4")
        market_sel = st.selectbox("상장시장", ["한국 전체","KOSPI","KOSDAQ","미국 전체","NASDAQ","일본 전체","Prime (Domestic Stocks)","Standard (Domestic Stocks)","Prime (Foreign Stocks)"], key="market_sel4")
        country_filter = market_filter = None
        if "전체" in market_sel:
            country_filter = market_sel.split()[0]
        else:
            country_filter = "한국" if market_sel in ("KOSPI","KOSDAQ") else "미국" if market_sel == "NASDAQ" else "일본"
            market_filter = market_sel
        class_type = st.radio("분류 체계", ["EMSEC","EMTEC"], horizontal=True, key="class_type4")
        if class_type == "EMSEC":
            sectors = sorted([s for s in DF_RAW.Sector.dropna().unique() if s != 'Unclassified'])
            sector_sel = st.selectbox("Sector", ["전체"] + sectors, key="sector_sel4")
            if sector_sel != "전체":
                indus = sorted(DF_RAW.loc[DF_RAW.Sector == sector_sel, "Industry"].dropna().unique())
                industry_sel = st.selectbox("Industry", ["전체"] + indus, key="industry_sel4")
            else:
                industry_sel = "전체"
            row_level = "Sector" if sector_sel == "전체" else "Industry" if industry_sel == "전체" else "Sub_industry"
        else:
            themes = sorted([t for t in DF_RAW.Theme.dropna().unique() if t != 'Unclassified'])
            theme_sel = st.selectbox("Theme", ["전체"] + themes, key="theme_sel4")
            if theme_sel != "전체":
                techs = sorted(DF_RAW.loc[DF_RAW.Theme == theme_sel, "Technology"].dropna().unique())
                tech_sel = st.selectbox("Technology", ["전체"] + techs, key="tech_sel4")
            else:
                tech_sel = "전체"
            row_level = "Theme" if theme_sel == "전체" else "Technology" if tech_sel == "전체" else "Sub_Technology"
        metric_base = {"시가총액": "Market Cap (2024-12-31)_USD", "자산총계": "Assets", "매출액": "Sales"}
        metric_name = st.selectbox("계측값", list(metric_base.keys()), key="metric_name4")
        metric_col = metric_base[metric_name]

    DF = DF_RAW[DF_RAW.Year == year_sel].copy()
    if country_filter: DF = DF[DF.Country == country_filter]
    if market_filter: DF = DF[DF.Market == market_filter]
    if class_type == "EMSEC":
        if sector_sel != "전체": DF = DF[DF.Sector == sector_sel]
        if industry_sel != "전체": DF = DF[DF.Industry == industry_sel]
    else:
        if theme_sel != "전체": DF = DF[DF.Theme == theme_sel]
        if tech_sel != "전체": DF = DF[DF.Technology == tech_sel]
    if metric_col not in DF.columns:
        st.error(f"'{metric_name}' 열이 없습니다. 데이터 파일을 확인해주세요.")
        st.stop()
    DF["metric_bil"] = DF[metric_col] / 1e9
    DF = DF[DF["metric_bil"].notna()]
    DF = DF[DF["metric_bil"] >= 0]
    if not DF.empty:
        max_th = DF["metric_bil"].quantile(0.999)
        DF = DF[DF["metric_bil"] <= max_th]
    if DF.empty:
        st.warning("조건에 맞는 데이터가 없습니다.")
        st.stop()

    country = DF['Country'].unique()[0] if len(DF['Country'].unique()) == 1 else 'Unclassified'
    currency = {'한국': 'KRW', '미국': 'USD', '일본': 'JPY', 'Unclassified': 'USD'}.get(country, 'USD')

    valid_vals = DF["metric_bil"]
    vl_max = valid_vals.max() if not valid_vals.empty else 0
    def make_edges(max_val: float) -> List[float]:
        base = [10,30,60,100,300,600]
        edges = [0]
        if max_val <= 0:
            edges += base[:1]
        else:
            exp = 0
            while True:
                factor = 10 ** exp
                for b in base:
                    edge = b * factor
                    if edge > max_val:
                        edges = sorted(set(edges))
                        return edges + [np.inf]
                    edges.append(edge)
                exp += 1
        edges = sorted(set(edges))
        return edges + [np.inf]
    bin_edges = make_edges(vl_max)
    bin_labels = ["0~"] + [f"{int(e):,}~" for e in bin_edges[1:-1]]
    DF["metric_bin"] = pd.cut(DF["metric_bil"], bins=bin_edges, labels=bin_labels, right=False)

    pivot = DF.groupby([row_level,"metric_bin"])["Company"].nunique().unstack(fill_value=0).reindex(columns=bin_labels, fill_value=0).astype(int)
    if pivot.empty:
        st.warning("조건에 맞는 데이터가 없어 집계표를 생성할 수 없습니다.")
        st.stop()
    subtotal = pd.DataFrame(pivot.sum()).T
    subtotal.index = ["Subtotal"]
    pivot_full = pd.concat([subtotal, pivot])

    rows = pivot_full.index.tolist()
    cols = pivot_full.columns.tolist()
    z_data = pivot_full.values
    z_max = np.nanmax(z_data)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z_data,
        x=cols,
        y=rows,
        colorscale="Greens",
        colorbar=dict(title="기업 수"),
        hovertemplate="%{y}/%{x}<br>기업수: %{z:,}<extra></extra>",
        xgap=1,
        ygap=1
    ))
    annotations = []
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            val = z_data[i, j]
            if val > 0:
                color = "white" if val > z_max * 0.5 else "black"
                annotations.append(
                    dict(
                        x=col,
                        y=row,
                        text=f"{int(val):,}",
                        showarrow=False,
                        font=dict(color=color, size=12)
                    )
                )
    fig.update_layout(
        annotations=annotations,
        height=max(600, 40*len(rows)),
        title=f"{metric_name} 분포 ({row_level}, {currency} billion)",
        xaxis_title="규모 범위",
        yaxis_title=row_level,
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"기업 수: {DF['Company'].nunique():,} | 통화: {currency}")
    with st.expander("📋 원본 집계표 보기", False):
        st.dataframe(pivot_full, use_container_width=True)
