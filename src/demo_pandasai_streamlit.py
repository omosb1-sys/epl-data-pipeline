import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
import os
import matplotlib.pyplot as plt

# 페이지 설정
st.set_page_config(layout="wide", page_title="K-League AI Analyst (PandasAI)")

st.title("🤖 K-League AI Data Analyst (Powered by PandasAI)")
st.markdown("""
**"Julius AI"의 로컬 파이썬 버전입니다.**  
데이터프레임에게 자연어로 질문하면, AI가 코드를 짜서 분석 결과와 차트를 보여줍니다.
""")

# 사이드바에서 API 키 입력 받기 (보안)
with st.sidebar:
    st.header("⚙️ 설정")
    
    # .env 파일에서 키 자동 로드 시도
    from dotenv import load_dotenv
    load_dotenv()
    default_key = os.getenv("OPENAI_API_KEY", "")
    
    api_key = st.text_input("OpenAI API Key를 입력하세요", value=default_key, type="password")
    if not api_key:
        st.info("💡 Tip: .env 파일에 OPENAI_API_KEY를 저장하면 매번 입력하지 않아도 됩니다.")
    else:
        st.success("API Key가 감지되었습니다.")

# 데이터 로드
DATA_PATH = 'data/raw/match_info.csv'

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 데이터 미리보기")
        st.dataframe(df.head())

    with col2:
        st.subheader("💬 AI 분석")
        
        if api_key:
            # SmartDataframe 초기화
            from pandasai.llm import OpenAI
            llm = OpenAI(api_token=api_key)
            sdf = SmartDataframe(df, config={"llm": llm})
            
            question = st.text_area("질문을 입력하세요:", placeholder="예: 홈 팀 득점 Top 5를 막대 그래프로 그려줘")
            
            if st.button("분석 실행"):
                if question:
                    with st.spinner("AI가 분석 코드를 작성하고 실행 중입니다..."):
                        try:
                            # PandasAI가 시각화를 생성할 때 Streamlit에 직접 표시되도록 처리
                            # (PandasAI 최신 버전은 이미지를 반환하거나 저장함)
                            response = sdf.chat(question)
                            
                            st.success("분석 완료!")
                            
                            # 응답 타입에 따라 처리
                            if isinstance(response, str) and os.path.exists(response) and response.endswith('.png'):
                                st.image(response) # 이미지 파일 경로인 경우
                            elif isinstance(response, (pd.DataFrame, pd.Series)):
                                st.dataframe(response)
                            else:
                                st.write(response)
                                
                        except Exception as e:
                            st.error(f"오류가 발생했습니다: {e}")
                else:
                    st.warning("질문을 입력해주세요.")
        else:
            st.warning("👈 왼쪽 사이드바에서 OpenAI API Key를 먼저 입력해주세요.")

else:
    st.error(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
