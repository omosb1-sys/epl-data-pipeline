"""
K-리그 AI 분석 대시보드 (Gemini 통합 버전)
GEMINI.md Protocol 준수 - 실시간 AI 인사이트 제공
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent))

# Gemini 분석 모듈 임포트
try:
    from gemini_k_league_analyst import GeminiKLeagueAnalyst
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("⚠️ Gemini 모듈을 불러올 수 없습니다. 기본 분석만 제공됩니다.")

# 페이지 설정
st.set_page_config(
    page_title="K-League AI Analysis Dashboard", 
    layout="wide",
    page_icon="⚽"
)

# 한글 폰트 설정 (Mac)
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 커스텀 CSS (심미적 엔지니어링)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# 타이틀
st.markdown('<h1 class="main-header">⚽ K-리그 AI 분석 대시보드</h1>', unsafe_allow_html=True)
st.markdown("""
**Gemini 2.0 Flash** 기반 실시간 전문가급 인사이트 제공  
*30년 차 시니어 분석가의 시각으로 데이터를 해석합니다*
""")

# 데이터베이스 연결
@st.cache_resource
def get_connection():
    db_path = Path(__file__).parent.parent / "data" / "processed" / "kleague.db"
    if not db_path.exists():
        st.error(f"❌ 데이터베이스를 찾을 수 없습니다: {db_path}")
        return None
    return sqlite3.connect(str(db_path), check_same_thread=False)

conn = get_connection()

# Gemini 분석가 초기화
@st.cache_resource
def get_analyst():
    if not GEMINI_AVAILABLE:
        return None
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.sidebar.warning("⚠️ GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        return None
    
    try:
        return GeminiKLeagueAnalyst(api_key=api_key)
    except Exception as e:
        st.sidebar.error(f"Gemini 초기화 실패: {str(e)}")
        return None

analyst = get_analyst()

# 사이드바 설정
st.sidebar.header("📊 분석 설정")

# 분석 모드 선택
analysis_mode = st.sidebar.radio(
    "분석 모드",
    ["📈 기본 통계", "🤖 AI 심층 분석"],
    help="AI 분석은 Gemini API 키가 필요합니다"
)

analysis_type = st.sidebar.selectbox(
    "분석 주제 선택", 
    [
        "팀별 실점 분석", 
        "홈/어웨이 득점 비교", 
        "시간대별 골 분포",
        "🆕 팀 성적 AI 분석",
        "🆕 라이벌 매치 예측",
        "🆕 리그 전체 트렌드"
    ]
)

# ============================================
# 기본 통계 분석
# ============================================

if analysis_mode == "📈 기본 통계":
    
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
            fig = px.bar(
                df_defense, 
                x='avg_goals_against', 
                y='team_name_ko', 
                orientation='h', 
                title='경기당 평균 실점 (낮을수록 우수)',
                color='avg_goals_against', 
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("### 📋 수비 랭킹")
            st.dataframe(
                df_defense.style.format({'avg_goals_against': '{:.2f}'}),
                height=600
            )
    
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
        
        fig = px.pie(
            df_home_away, 
            values='avg_score', 
            names='location', 
            title='평균 득점 비중 (홈 vs 어웨이)',
            color_discrete_sequence=['#667eea','#764ba2'],
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        home_advantage = ((df_home_away.iloc[0]['avg_score']/df_home_away.iloc[1]['avg_score'])-1)*100
        st.markdown(f"""
        <div class="insight-box">
            <h3>🏠 홈 어드밴티지 분석</h3>
            <p style="font-size: 1.2rem;">
            홈팀 평균 득점이 어웨이팀보다 약 <strong>{home_advantage:.1f}%</strong> 높습니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
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
        
        fig = px.histogram(
            df_goals, 
            x='match_min', 
            nbins=18, 
            title='경기 시간대별 득점 빈도 (5분 단위)',
            labels={'match_min': '경기 시간(분)'},
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 경기 막판(75분 이후)에 득점이 가장 많이 발생하는 경향을 확인할 수 있습니다.")

# ============================================
# AI 심층 분석 (Gemini)
# ============================================

elif analysis_mode == "🤖 AI 심층 분석":
    
    if not analyst:
        st.error("""
        ❌ **Gemini API 키가 설정되지 않았습니다.**
        
        다음 명령어로 API 키를 설정하세요:
        ```bash
        export GEMINI_API_KEY="your-api-key-here"
        ```
        
        API 키 발급: https://makersuite.google.com/app/apikey
        """)
        st.stop()
    
    if analysis_type == "🆕 팀 성적 AI 분석":
        st.header("🤖 팀 성적 AI 심층 분석")
        
        # 팀 목록 가져오기
        teams_query = "SELECT DISTINCT home_team_name_ko FROM match_info ORDER BY home_team_name_ko"
        teams = pd.read_sql(teams_query, conn)['home_team_name_ko'].tolist()
        
        selected_team = st.selectbox("분석할 팀 선택", teams)
        
        if st.button("🚀 AI 분석 시작", key="team_analysis"):
            with st.spinner(f"🧠 Gemini가 {selected_team} 데이터를 분석 중..."):
                # 팀 데이터 준비
                query = f"""
                SELECT 
                    game_id,
                    CASE 
                        WHEN home_team_name_ko = '{selected_team}' THEN home_score
                        ELSE away_score
                    END as 득점,
                    CASE 
                        WHEN home_team_name_ko = '{selected_team}' THEN away_score
                        ELSE home_score
                    END as 실점,
                    CASE
                        WHEN (home_team_name_ko = '{selected_team}' AND home_score > away_score) OR
                             (away_team_name_ko = '{selected_team}' AND away_score > home_score) THEN 1
                        ELSE 0
                    END as 승,
                    CASE
                        WHEN home_score = away_score THEN 1
                        ELSE 0
                    END as 무,
                    CASE
                        WHEN (home_team_name_ko = '{selected_team}' AND home_score < away_score) OR
                             (away_team_name_ko = '{selected_team}' AND away_score < home_score) THEN 1
                        ELSE 0
                    END as 패
                FROM match_info
                WHERE home_team_name_ko = '{selected_team}' OR away_team_name_ko = '{selected_team}'
                """
                df_team = pd.read_sql(query, conn)
                df_team['팀명'] = selected_team
                
                # AI 분석 실행
                result = analyst.analyze_team_performance(df_team, selected_team)
                
                if 'error' in result:
                    st.error(f"❌ {result['error']}")
                else:
                    # 분석 결과 표시
                    st.markdown(f"""
                    <div class="insight-box">
                        <h2>🎯 {selected_team} AI 분석 리포트</h2>
                        <p><em>분석 시간: {result['timestamp']}</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(result['analysis'])
                    
                    # 기초 통계 표시
                    with st.expander("📊 기초 통계 데이터"):
                        col1, col2, col3, col4 = st.columns(4)
                        stats = result['stats']
                        
                        col1.metric("총 경기", stats['총 경기수'])
                        col2.metric("승률", f"{stats['승률']}%")
                        col3.metric("득점", stats['득점'])
                        col4.metric("실점", stats['실점'])
                    
                    # 리포트 저장
                    if st.button("💾 리포트 저장"):
                        filepath = analyst.save_report(result)
                        st.success(f"✅ 리포트 저장 완료: {filepath}")
    
    elif analysis_type == "🆕 라이벌 매치 예측":
        st.header("⚔️ 라이벌 매치 AI 예측")
        
        teams_query = "SELECT DISTINCT home_team_name_ko FROM match_info ORDER BY home_team_name_ko"
        teams = pd.read_sql(teams_query, conn)['home_team_name_ko'].tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("팀 1", teams, key="team1")
        with col2:
            team2 = st.selectbox("팀 2", teams, key="team2", index=1 if len(teams) > 1 else 0)
        
        if st.button("🔮 승부 예측", key="match_prediction"):
            if team1 == team2:
                st.warning("⚠️ 다른 팀을 선택해주세요.")
            else:
                with st.spinner(f"🧠 Gemini가 {team1} vs {team2} 매치를 분석 중..."):
                    # 각 팀 데이터 준비
                    query = """
                    SELECT 
                        CASE 
                            WHEN home_team_name_ko = ? THEN home_team_name_ko
                            ELSE away_team_name_ko
                        END as 팀명,
                        CASE 
                            WHEN home_team_name_ko = ? THEN home_score
                            ELSE away_score
                        END as 득점,
                        CASE 
                            WHEN home_team_name_ko = ? THEN away_score
                            ELSE home_score
                        END as 실점,
                        CASE
                            WHEN (home_team_name_ko = ? AND home_score > away_score) OR
                                 (away_team_name_ko = ? AND away_score > home_score) THEN 1
                            ELSE 0
                        END as 승
                    FROM match_info
                    WHERE home_team_name_ko = ? OR away_team_name_ko = ?
                    """
                    
                    df1 = pd.read_sql(query, conn, params=[team1]*5 + [team1, team1])
                    df2 = pd.read_sql(query, conn, params=[team2]*5 + [team2, team2])
                    df_combined = pd.concat([df1, df2])
                    
                    result = analyst.compare_teams(df_combined, team1, team2)
                    
                    if 'error' in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.markdown(f"""
                        <div class="insight-box">
                            <h2>⚔️ {team1} vs {team2}</h2>
                            <p><em>AI 승부 예측 리포트</em></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(result['analysis'])
                        
                        # 비교 통계
                        with st.expander("📊 팀 비교 통계"):
                            comp_df = pd.DataFrame(result['comparison']).T
                            st.dataframe(comp_df.style.format("{:.2f}"))
    
    elif analysis_type == "🆕 리그 전체 트렌드":
        st.header("🏆 K-리그 전체 트렌드 AI 분석")
        
        if st.button("🚀 리그 분석 시작", key="league_analysis"):
            with st.spinner("🧠 Gemini가 리그 전체 데이터를 분석 중..."):
                # 전체 리그 데이터 준비
                query = """
                SELECT 
                    team_name_ko as 팀명,
                    SUM(wins) as 승,
                    SUM(draws) as 무,
                    SUM(losses) as 패,
                    SUM(goals_for) as 득점,
                    SUM(goals_against) as 실점
                FROM (
                    SELECT 
                        home_team_name_ko as team_name_ko,
                        CASE WHEN home_score > away_score THEN 1 ELSE 0 END as wins,
                        CASE WHEN home_score = away_score THEN 1 ELSE 0 END as draws,
                        CASE WHEN home_score < away_score THEN 1 ELSE 0 END as losses,
                        home_score as goals_for,
                        away_score as goals_against
                    FROM match_info
                    UNION ALL
                    SELECT 
                        away_team_name_ko as team_name_ko,
                        CASE WHEN away_score > home_score THEN 1 ELSE 0 END as wins,
                        CASE WHEN home_score = away_score THEN 1 ELSE 0 END as draws,
                        CASE WHEN away_score < home_score THEN 1 ELSE 0 END as losses,
                        away_score as goals_for,
                        home_score as goals_against
                    FROM match_info
                )
                GROUP BY team_name_ko
                """
                df_league = pd.read_sql(query, conn)
                
                result = analyst.generate_league_overview(df_league)
                
                if 'error' in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.markdown("""
                    <div class="insight-box">
                        <h2>🏆 K-리그 전체 트렌드 분석</h2>
                        <p><em>시즌 전체 흐름 및 파워 랭킹</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(result['analysis'])
                    
                    # 순위표 시각화
                    with st.expander("📊 팀별 순위표"):
                        rankings_df = pd.DataFrame(result['rankings'])
                        st.dataframe(rankings_df.style.format({'승률': '{:.2f}%'}))

# 푸터
st.sidebar.markdown("---")
st.sidebar.markdown("""
**🤖 Powered by**
- Gemini 2.0 Flash (Experimental)
- Streamlit
- SQLite

**📖 GEMINI.md Protocol v1.9**

*Developed by Antigravity AI*
""")

# 디버그 정보
if st.sidebar.checkbox("🔧 디버그 정보"):
    st.sidebar.json({
        "Gemini Available": GEMINI_AVAILABLE,
        "Analyst Initialized": analyst is not None,
        "DB Connection": conn is not None,
        "Analysis Mode": analysis_mode,
        "Analysis Type": analysis_type
    })
