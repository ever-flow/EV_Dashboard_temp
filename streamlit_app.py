import config  # 환경 설정 및 상수

import streamlit as st

from data_processing import load_processed_data
from visualizations.heatmap import show_heatmap
from visualizations.scatter import show_scatter
from visualizations.growth import show_growth_pattern
from visualizations.scale import show_scale_heatmap

# Streamlit Page Config
st.set_page_config(page_title="Integrated EMSEC × EMTEC Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

DF_RAW = load_processed_data()

st.sidebar.title("시각화 선택")
visualization = st.sidebar.radio("선택하세요:", ["히트맵", "점도표", "성장 패턴", "규모 변수"])

if visualization == "히트맵":
    show_heatmap(DF_RAW)
elif visualization == "점도표":
    show_scatter(DF_RAW)
elif visualization == "성장 패턴":
    show_growth_pattern(DF_RAW)
else:
    show_scale_heatmap(DF_RAW)
