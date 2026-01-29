
# 📊 SKILL: Interactive Dashboard Architect (Streamlit/Pygwalker)

> **"Stop sending static PPTs."**  
> 데이터 분석 결과를 클릭 가능한 **인터랙티브 웹 앱(Streamlit)**으로 즉시 변환하는 아키텍처 가이드입니다.

## 1. UX Assessment (사용자 경험 원칙)
*   **3-Second Rule**: 앱 로딩 후 3초 안에 핵심 지표(KPI)가 보여야 한다. (무거운 차트는 Lazy Loading)
*   **Interactive First**: 정적 이미지(`plt.show`) 대신 `plotly`, `altair`, `pygwalker` 등 마우스 오버가 가능한 라이브러리를 사용한다.
*   **Mobile Friendly**: `st.columns` 사용 시 모바일 화면 깨짐을 고려한다.

## 2. Performance Architecture (성능 최적화)
*   **Cache Everything**: 데이터 로드 함수에는 반드시 `@st.cache_data`, 모델 로드에는 `@st.cache_resource`를 붙인다.
*   **PyGWalker Optimization**: 탐색적 분석(EDA) 툴인 PyGWalker 사용 시 `kernel_computation=True`를 켜서 렌더링 속도를 높이고, 렌더러 객체를 캐싱한다.
*   **Orjson Speed-up**: JSON 직렬화 속도 향상을 위해 `orjson` 라이브러리를 활용한다.

## 3. Code Snippet (Dashboard Template)

### 3.1 High-Performance Streamlit Structure
```python
import streamlit as st
import polars as pl
import plotly.express as px
from pygwalker.api.streamlit import StreamlitRenderer

# 1. Config
st.set_page_config(layout="wide", page_title="Insight Dashboard")

# 2. Cached Data Loader (With Polars)
@st.cache_data
def load_data():
    return pl.scan_parquet("data.parquet").collect().to_pandas()

df = load_data()

# 3. Layout Strategy (Metric First)
c1, c2, c3 = st.columns(3)
c1.metric("Total Goals", df['goals'].sum())
c2.metric("Avg xG", df['xg'].mean().round(2))

# 4. Interactive Chart (Plotly)
st.subheader("Trends")
fig = px.line(df, x='date', y='goals', color='team')
st.plotly_chart(fig, use_container_width=True)

# 5. Explorer Mode (PyGWalker - Cached)
@st.cache_resource
def get_pyg_renderer(data):
    return StreamlitRenderer(data, spec="./gw_config.json", spec_io_mode="RW")

if st.toggle("Show Explorer"):
    renderer = get_pyg_renderer(df)
    renderer.explorer()
```
