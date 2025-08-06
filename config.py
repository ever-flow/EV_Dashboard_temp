import os, pathlib, tempfile

# 1) HOME이 없거나 '/'면 강제로 교체
if os.environ.get("HOME", "/") in ("", "/"):
    os.environ["HOME"] = "/tmp"             # 필요하면 tempfile.mkdtemp() 사용

# 2) .streamlit 디렉터리 생성
streamlit_dir = pathlib.Path(os.environ["HOME"], ".streamlit")
streamlit_dir.mkdir(parents=True, exist_ok=True)

# 3) 사용 통계 비활성화
os.environ["STREAMLIT_DISABLE_USAGE_STATS"] = "true"

PERIODS = ["LTM", "LTM-1", "LTM-2", "LTM-3"]
METHODS = ["AVG", "MED", "HRM", "AGG"]
COLORS = {"AVG": "#1f77b4", "MED": "#ff7f0e", "HRM": "#2ca02c", "AGG": "#d62728"}
EXCHANGE_RATES = {  # 2024‑12‑31 기준 환율 (원/달러, 엔/달러)
    '한국': 1380.0,
    '미국': 1.0,
    '일본': 157.0,
    'Unclassified': 1.0
}
