import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

# 스타일링 (Premium Dark Mode)
st.set_page_config(page_title="Antigravity Insight Center", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #161B22; padding: 20px; border-radius: 10px; border: 1px solid #30363D; }
    .header-text { font-size: 2.2rem; font-weight: 800; color: #58A6FF; margin-bottom: 2rem; }
    .skill-card { background-color: #21262D; padding: 15px; border-radius: 8px; border-left: 5px solid #238636; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-text">🧠 Antigravity Intelligence Insight Center</div>', unsafe_allow_html=True)

LOG_FILE = os.path.join(os.path.dirname(__file__), "internal", "session_history.jsonl")

if not os.path.exists(LOG_FILE):
    st.info("📊 아직 분석된 세션 데이터가 없습니다. Antigravity와 대화를 시작하면 기록이 쌓입니다.")
else:
    df = pd.read_json(LOG_FILE, lines=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 상단 메트릭
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 활동 건수", len(df))
    with col2:
        st.metric("주요 태스크", df[df['type'] == 'ANALYSIS'].shape[0] if 'ANALYSIS' in df['type'].values else 0)
    with col3:
        st.metric("시스템 최적화", df[df['type'] == 'OPTIMIZE'].shape[0] if 'OPTIMIZE' in df['type'].values else 0)
    with col4:
        st.metric("마지막 활동", df['timestamp'].iloc[-1].strftime("%H:%M:%S"))

    # 활동 시계열 차트
    st.subheader("🗓️ 활동 타임라인")
    df['hour'] = df['timestamp'].dt.hour
    fig = px.histogram(df, x='timestamp', color='type', template="plotly_dark", barmode='group')
    st.plotly_chart(fig, use_container_width=True)

    # 지능형 제언 (Skill Factory Preview)
    st.subheader("🚀 자동화 제언 (AI-Driven Patterns)")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="skill-card">
            <h4>✓ 데이터 수집 자동 복구 패턴 감지</h4>
            <p>최근 3회 이상 <code>collect_data.py</code> 에러를 AI가 수정했습니다. <br>
            이를 <b>Daily-Auto-Heal</b> 워크플로우로 등록할까요?</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="skill-card" style="border-left-color: #8957E5;">
            <h4>✓ 대용량 SQL 최적화 스킬 발견</h4>
            <p>DuckDB를 활용한 패턴이 반복되고 있습니다. <br>
            <code>fast_sql_query.md</code> 스킬 생성을 추천합니다.</p>
        </div>
        """, unsafe_allow_html=True)

    # 상세 기록
    with st.expander("📝 전체 세션 로그 확인"):
        st.dataframe(df.sort_values('timestamp', ascending=False), use_container_width=True)

st.caption(f"⏱️ 마지막 분석 갱신: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
