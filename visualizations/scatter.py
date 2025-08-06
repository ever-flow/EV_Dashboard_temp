import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config import PERIODS, METHODS, COLORS


def show_scatter(DF_RAW):
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
