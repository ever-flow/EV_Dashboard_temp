# EMSEC × EMTEC 대시보드

이 저장소는 전기차 섹터 데이터를 다양한 시각화로 탐색할 수 있는
[Streamlit](https://streamlit.io) 기반 대시보드입니다. 기존의 단일 스크립트
`streamlit_app.py`를 기능별 모듈로 분리하여 유지보수성과 재사용성을 높였습니다.

## 프로젝트 구조

```
config.py                # 전역 설정, 환율, 색상 등 공통 상수
data_utils.py            # 통화 변환, 분류 체계 파싱, 보조 유틸리티
data_processing.py       # 원천 데이터 로드 및 전처리
aggregation.py           # 히트맵 집계 로직
visualizations/          # 시각화 모듈 (heatmap, scatter, growth, scale)
streamlit_app.py         # Streamlit 진입점
```

## 설치 방법

1. Python 3.10 이상이 설치되어 있어야 합니다.
2. 필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
streamlit run streamlit_app.py
```

웹 브라우저에서 안내된 주소로 접속하면 대시보드를 사용할 수 있습니다.

## 데이터

- `heatmap_data_with_SE_v2.xlsx` 파일을 사용하며, 프로젝트 루트에 위치해야 합니다.

## 라이선스

이 프로젝트는 별도의 명시가 없는 한 자유롭게 활용할 수 있습니다.

