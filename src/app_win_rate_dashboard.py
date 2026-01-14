import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# 프로젝트 루트를 경로에 추가 (KLeagueForecaster 임포트용)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.k_league_timesfm_forecast import KLeagueForecaster

# ==========================================
# 1. 환경 설정 및 데이터 로드
# ==========================================
st.set_page_config(page_title="K-League AI Win Rate Dashboard", layout="wide", page_icon="🚀")

PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data/processed/team_match_results.csv")

@st.cache_data
def load_data():
    if not os.path.exists(PROCESSED_DATA_PATH):
        st.error(f"데이터 파일을 찾을 수 없습니다: {PROCESSED_DATA_PATH}. 전처리(Chunk 1)를 먼저 실행해주세요.")
        return None
    return pd.read_csv(PROCESSED_DATA_PATH)

df = load_data()

# ==========================================
# 2. 사이드바 - 필터 (Sidebar)
# ==========================================
if df is not None:
    st.sidebar.title("🔍 검색 옵션")
    
    # 시즌 선택
    seasons = sorted(df['season_id'].unique())
    selected_seasons = st.sidebar.multiselect("📅 시즌 선택", seasons, default=seasons)
    
    # 팀 선택
    teams = sorted(df['team'].unique())
    selected_teams = st.sidebar.multiselect("⚽ 팀 선택", teams, default=teams)
    
    # 필터링 적용
    mask = df['season_id'].isin(selected_seasons) & df['team'].isin(selected_teams)
    filtered_df = df[mask]

    # ==========================================
    # 3. 데이터 집계 (Win Rate Logic)
    # ==========================================
    def get_team_stats(data):
        stats = data.groupby('team').agg(
            Total_Games=('game_id', 'count'),
            Wins=('result', lambda x: (x == 'Win').sum()),
            Draws=('result', lambda x: (x == 'Draw').sum()),
            Losses=('result', lambda x: (x == 'Loss').sum()),
            Points=('points', 'sum')
        ).reset_index()
        
        stats['Win_Rate'] = (stats['Wins'] / stats['Total_Games'] * 100).round(1)
        return stats.sort_values(by='Win_Rate', ascending=False)

    team_stats = get_team_stats(filtered_df)

    # ==========================================
    # 4. 메인 대시보드 UI
    # ==========================================
    st.title("🏆 K-League AI 분석 대시보드")
    st.markdown("---")

    # 탭 구성: 과거 데이터 분석 vs AI 딥러닝 예측
    tab1, tab2 = st.tabs(["📊 과거 승률 분석", "🚀 AI 딥러닝 예측 (Deep Learning)"])

    with tab1:
        if not team_stats.empty:
            # KPI 카드
            c1, c2, c3, c4 = st.columns(4)
            top_team = team_stats.iloc[0]
            
            c1.metric("총 경기 수", f"{filtered_df['game_id'].nunique()} 경기")
            c2.metric("참여 구단", f"{team_stats['team'].nunique()} 개")
            c3.metric("최고 승률 팀", f"{top_team['team']}")
            c4.metric("최고 승률", f"{top_team['Win_Rate']}%")

            st.markdown("### 📊 구단별 승률 랭킹")
            
            fig_win_rate = px.bar(
                team_stats, 
                x='team', 
                y='Win_Rate',
                text='Win_Rate',
                color='Win_Rate',
                color_continuous_scale='Viridis',
                labels={'Win_Rate': '승률 (%)', 'team': '팀명'}
            )
            fig_win_rate.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_win_rate.update_layout(xaxis={'categoryorder':'total descending'}, height=500)
            st.plotly_chart(fig_win_rate, use_container_width=True)

            # 상세 데이터 테이블
            with st.expander("📝 과거 상세 통계 보기"):
                st.dataframe(team_stats, use_container_width=True)
        else:
            st.warning("데이터가 없습니다.")

    with tab2:
        st.markdown("### 🤖 신경망(Neural Network) 기반 차기 화력 예측")
        st.info("시니어 분석가 가이드: 본 예측은 단순 과거 평균이 아닌, 최근 4주간의 득점 패턴을 MLP 신경망이 학습한 '가속도'를 반영한 결과입니다.")
        
        if st.button("AI 딥러닝 분석 엔진 가동"):
            with st.spinner("최신 흐름 학습 중..."):
                # 딥러닝 예보관 호출
                forecaster = KLeagueForecaster(data_path=os.path.join(BASE_DIR, 'data/raw/match_info.csv'))
                forecast_report = forecaster.run_league_analysis()
                
                if forecast_report is not None:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig_forecast = px.bar(
                            forecast_report,
                            x='구단',
                            y='예상_주간_득점력',
                            color='예상_주간_득점력',
                            color_continuous_scale='Reds',
                            title="AI 차기 라운드 득점 화력 예측"
                        )
                        st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### 🏆 화력 TOP 5")
                        st.table(forecast_report.head(5)[['구단', '예상_주간_득점력']])
                        
                    st.success("분석이 완료되었습니다. 이 수치가 높을수록 해당 팀의 최근 기세가 매우 강력함을 의미합니다.")
                else:
                    st.error("분석 데이터를 생성할 수 없습니다.")

else:
    st.error("데이터 로드에 실패했습니다.")
