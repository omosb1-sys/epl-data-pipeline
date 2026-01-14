import streamlit as st
import pandas as pd
import os
import pygwalker as pyg
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# 페이지 기본 설정
st.set_page_config(layout="wide", page_title="K-League Unified Analyst")

# 스타일링
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1E88E5;}
    .sub-header {font-size: 1.5rem; color: #424242;}
</style>
""", unsafe_allow_html=True)

# 데이터 로드 (캐싱하여 성능 최적화)
@st.cache_data
def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

DATA_PATH = 'data/raw/match_info.csv'
df = load_data(DATA_PATH)

# 사이드바: 모드 선택 및 설정
with st.sidebar:
    st.title("⚽️ K-League Studio")
    st.markdown("---")
    
    # 분석 모드 선택
    mode = st.radio(
        "분석 도구 선택",
        ["🔍 Drag & Drop 탐색 (PyGWalker)", "🤖 AI 질의응답 (PandasAI)"],
        index=0
    )
    
    st.markdown("---")
    st.caption("2026 AI Data Analyst Toolkit")

# 메인 콘텐츠 영역
if df is not None:
    if mode == "🔍 Drag & Drop 탐색 (PyGWalker)":
        st.markdown('<div class="main-header">K-League Interactive Explorer</div>', unsafe_allow_html=True)
        st.markdown('Bricks 스타일의 **드래그 앤 드롭 시각화** 도구입니다. 변수를 X/Y축으로 끌어당기세요.')
        
        # PyGWalker 실행
        pyg_html = pyg.to_html(df)
        st.components.v1.html(pyg_html, height=1000, scrolling=True)

    elif mode == "🤖 AI 질의응답 (PandasAI)":
        st.markdown('<div class="main-header">K-League AI Analyst</div>', unsafe_allow_html=True)
        st.markdown('Julius AI 스타일의 **자연어 분석** 도구입니다. 궁금한 것을 말로 물어보세요.')
        
        # PandasAI 설정
        from pandasai import SmartDataframe
        from pandasai.llm import OpenAI # Fixed Import
        
        # API Key 로드
        load_dotenv()
        env_key = os.getenv("OPENAI_API_KEY", "")
        
        # 모델 선택 (친절한 이름으로 변경)
        model_map = {
            "☁️ OpenAI (온라인/유료)": "OpenAI (Cloud)",
            "✅ Llama 3.2 (설치완료/즉시사용)": "llama3.2:latest",
            "✅ Microsoft Phi-3.5 (설치완료/성능우수)": "phi3.5:latest",
            "✅ Qwen 2.5 (설치완료/코딩)": "qwen2.5:3b",
            "✅ Mistral (설치완료/범용)": "mistral:latest"
        }
        
        selected_display = st.selectbox(
            "🧠 분석 모델 선택 (오프라인 모드 지원)", 
            list(model_map.keys()),
            index=2 # 기본값을 설치된 Phi-3.5로 변경
        )
        
        llm_choice = model_map[selected_display]
        # model_options = ["OpenAI (Cloud)", "phi3:medium", "gemma2:9b", "qwen2.5-coder:7b"]
        # llm_choice = st.selectbox("🧠 분석 모델 선택", model_options)
        
        # OpenAI 선택 시에만 키 입력창 노출
        api_key = env_key
        if llm_choice == "OpenAI (Cloud)":
            with st.expander("🔑 AI 설정 (API Key)", expanded=not env_key):
                 api_key = st.text_input("OpenAI API Key", value=env_key, type="password")
        else:
            # 로컬 모델은 API 키 불필요
            api_key = "local_ollama"

        if api_key:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("데이터 미리보기")
                st.dataframe(df.head(10), use_container_width=True)
                
            with col2:
                # LLM 설정
                if llm_choice == "OpenAI (Cloud)":
                    llm = OpenAI(api_token=api_key)
                else: 
                    # 로컬 Ollama 모델 사용
                    from langchain_community.llms import Ollama
                    llm = Ollama(model=llm_choice)
                
                sdf = SmartDataframe(df, config={"llm": llm, "enable_cache": False})
                
                st.subheader("💬 무엇이 궁금하신가요?")
                question = st.text_area("질문 입력", height=100, placeholder="예: 구단별 평균 득점을 막대 그래프로 보여줘")
                
                if st.button("🚀 분석 실행", type="primary"):
                    if question:
                        with st.spinner(f"🧠 {llm_choice} 모델이 데이터를 분석 중입니다..."):
                            try:
                                response = sdf.chat(question)
                                
                                st.success("분석 완료!")
                                # 결과 유형에 따른 렌더링
                                if isinstance(response, str) and (response.endswith('.png') or os.path.isfile(response)):
                                    st.image(response)
                                elif isinstance(response, (pd.DataFrame, pd.Series)):
                                    st.dataframe(response)
                                else:
                                    st.write(response)
                                    
                            except Exception as e:
                                st.error(f"분석 중 오류 발생: {e}")
                    else:
                        st.warning("질문을 입력해주세요.")
        else:
            if llm_choice == "OpenAI (Cloud)":
                st.warning("AI 분석을 사용하려면 OpenAI API Key가 필요합니다. (.env 파일 또는 위 입력창 이용)")
            else:
                 st.info(f"선택하신 {llm_choice} 모델이 로컬에 설치되어 있어야 작동합니다.")

else:
    st.error(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
