import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List


def show_scale_heatmap(DF_RAW):
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
