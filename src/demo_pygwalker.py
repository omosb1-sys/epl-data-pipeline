import pandas as pd
import pygwalker as pyg
import streamlit as st
import os

# 페이지 설정 (전체 화면 활용)
st.set_page_config(layout="wide", page_title="K-League Interactive Analysis")

st.title("⚽️ K-League Interactive Dashboard (Powered by PyGWalker)")
st.markdown("""
**"Bricks"와 같은 드래그 앤 드롭 분석 도구입니다.**  
데이터를 외부로 보낼 필요 없이, Python 환경 내에서 Tableau처럼 자유롭게 시각화하세요.
""")

# 데이터 로드
DATA_PATH = 'data/raw/match_info.csv'
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    
    # 탭 구성
    tab1, tab2 = st.tabs(["📊 드래그 앤 드롭 분석 (PyGWalker)", "💾 원본 데이터"])
    
    with tab1:
        st.write("아래 공간에서 변수를 드래그하여 차트를 만들어보세요.")
        # PyGWalker를 Streamlit HTML로 변환하여 렌더링
        pyg_html = pyg.to_html(df)
        st.components.v1.html(pyg_html, height=1000, scrolling=True)
        
    with tab2:
        st.dataframe(df)

else:
    st.error("데이터 파일을 찾을 수 없습니다.")
