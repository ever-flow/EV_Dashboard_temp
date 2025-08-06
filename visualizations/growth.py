import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from typing import List


def show_growth_pattern(DF_RAW):
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

