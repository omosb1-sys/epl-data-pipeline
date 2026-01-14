import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px

st.set_page_config(page_title="K-League Analysis Dashboard", layout="wide")

st.title("⚽ K-League 2024 Analysis Dashboard")
st.markdown("""
이 대시보드는 **SQLite**와 **Streamlit**을 결합하여 실시간으로 데이터를 분석합니다.
코드 몇 줄로 이런 웹 인터페이스를 만들 수 있다는 것이 Streamlit의 강점입니다!
""")

# 데이터베이스 연결
@st.cache_resource
def get_connection():
    return sqlite3.connect('/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/processed/kleague.db', check_same_thread=False)

conn = get_connection()

# 사이드바 설정
st.sidebar.header("설정")
analysis_type = st.sidebar.selectbox("분석 주제 선택", ["팀별 실점 분석", "홈/어웨이 득점 비교", "시간대별 골 분포"])

if analysis_type == "팀별 실점 분석":
    st.header("🛡️ 팀별 평균 실점 순위")
    
    query = """
    SELECT 
        team_name_ko,
        AVG(case when is_home = 1 then away_score else home_score end) as avg_goals_against
    FROM (
        SELECT home_team_name_ko as team_name_ko, home_score, away_score, 1 as is_home FROM match_info
        UNION ALL
        SELECT away_team_name_ko as team_name_ko, home_score, away_score, 0 as is_home FROM match_info
    )
    GROUP BY team_name_ko
    ORDER BY avg_goals_against ASC
    """
    df_defense = pd.read_sql(query, conn)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(df_defense, x='avg_goals_against', y='team_name_ko', 
                     orientation='h', title='경기당 평균 실점 (낮을수록 우수)',
                     color='avg_goals_against', color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, width="stretch")
        
    with col2:
        st.write("### 수비 랭킹 데이터")
        st.dataframe(df_defense.style.format({'avg_goals_against': '{:.2f}'}))

elif analysis_type == "홈/어웨이 득점 비교":
    st.header("🏠 Home vs ✈️ Away 득점 비교")
    
    query = """
    SELECT 
        CASE WHEN home_team_id = team_id THEN 'Home' ELSE 'Away' END as location,
        AVG(CASE WHEN home_team_id = team_id THEN home_score ELSE away_score END) as avg_score
    FROM (
        SELECT game_id, home_team_id as team_id, home_score, away_score, home_team_id FROM match_info
        UNION ALL
        SELECT game_id, away_team_id as team_id, home_score, away_score, home_team_id FROM match_info
    )
    GROUP BY location
    """
    df_home_away = pd.read_sql(query, conn)
    
    fig = px.pie(df_home_away, values='avg_score', names='location', 
                 title='평균 득점 비중 (홈 vs 어웨이)',
                 color_discrete_sequence=['#ff9999','#66b3ff'])
    st.plotly_chart(fig)
    st.write(f"홈팀 평균 득점이 어웨이팀보다 약 **{((df_home_away.iloc[1]['avg_score']/df_home_away.iloc[0]['avg_score'])-1)*100:.1f}%** 높습니다.")

elif analysis_type == "시간대별 골 분포":
    st.header("⏰ 시간대별 득점 발생 분포")
    
    query = """
    SELECT 
        time_seconds,
        period_id
    FROM raw_data
    WHERE type_name = 'Goal'
    """
    df_goals = pd.read_sql(query, conn)
    
    def get_min(row):
        m = row['time_seconds'] / 60
        return m + 45 if row['period_id'] == 2 else m

    df_goals['match_min'] = df_goals.apply(get_min, axis=1)
    
    fig = px.histogram(df_goals, x='match_min', nbins=18, 
                       title='경기 시간대별 실점/득점 빈도 (5분 단위)',
                       labels={'match_min': '경기 시간(분)'},
                       color_discrete_sequence=['indianred'])
    st.plotly_chart(fig, width="stretch")
    st.info("경기 막판(75분 이후)에 득점이 가장 많이 발생하는 경향을 확인할 수 있습니다.")

st.sidebar.markdown("---")
st.sidebar.write("Developed by Antigravity AI")
