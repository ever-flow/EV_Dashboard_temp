import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config import PERIODS
from aggregation import compute_aggregate


def show_heatmap(DF_RAW):
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


