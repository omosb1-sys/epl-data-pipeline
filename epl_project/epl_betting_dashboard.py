import streamlit as st
import pandas as pd
import time
from datetime import datetime
from epl_duckdb_manager import EPLDuckDBManager
from epl_realtime_ingestor import EPLRealtimeIngestor
from epl_prediction_engine import EPLPredictionEngine

# 페이지 설정
st.set_page_config(page_title="EPL Real-time Value Betting AI", page_icon="⚽", layout="wide")

# 스타일 설정
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1a1c23; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .value-bet { color: #00ff00; font-weight: bold; border: 2px solid #00ff00; padding: 10px; border-radius: 5px; text-align: center; }
    .no-bet { color: #888; text-align: center; font-style: italic; }
    </style>
""", unsafe_allow_html=True)

# 초기화
@st.cache_resource
def init_system():
    db = EPLDuckDBManager()
    ingestor = EPLRealtimeIngestor()
    engine = EPLPredictionEngine()
    return db, ingestor, engine

db, ingestor, engine = init_system()

st.title("⚽ EPL Real-time Prediction Dashboard")
st.subheader("AI-Powered Live Odds & Value Betting Tracker")

# 사이드바: 컨트롤
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    if st.button("🔄 Force Data Refresh"):
        with st.spinner("Updating API Data..."):
            ingestor.run_ingestion_loop()
            engine.run_all_predictions()
            st.success("Updated!")
            
    auto_refresh = st.checkbox("Auto Refresh (60s)", value=True)
    st.info("💡 LinkedIn Idea: RapidAPI + DuckDB + XGBoost + Optuna")

# 데이터 로드
def load_dashboard_data():
    fixtures = db.conn.execute("""
        SELECT f.*, p.home_win_prob, p.draw_prob, p.away_win_prob, p.value_bet_side, p.value_bet_edge,
               o.home_win_odds, o.draw_odds, o.away_win_odds
        FROM fixtures f
        LEFT JOIN predictions p ON f.fixture_id = p.fixture_id
        LEFT JOIN (
            SELECT fixture_id, home_win_odds, draw_odds, away_win_odds
            FROM odds
            QUALIFY ROW_NUMBER() OVER (PARTITION BY fixture_id ORDER BY timestamp DESC) = 1
        ) o ON f.fixture_id = o.fixture_id
        WHERE f.status != 'Finished'
        ORDER BY f.date ASC
    """).df()
    return fixtures

data = load_dashboard_data()

if data.empty:
    st.warning("No live or upcoming fixtures found. Try 'Force Data Refresh'.")
else:
    # 대시보드 메인 그리드
    for idx, row in data.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([2, 3, 2])
            
            with col1:
                st.markdown(f"### {row['home_team']} vs {row['away_team']}")
                st.write(f"📅 {row['date']}")
                st.write(f"🏟️ {row['venue']}")
                st.caption(f"Status: {row['status']}")
            
            with col2:
                st.markdown("#### 🤖 AI 승률 예측")
                p_home = row['home_win_prob'] if not pd.isna(row['home_win_prob']) else 0.33
                p_draw = row['draw_prob'] if not pd.isna(row['draw_prob']) else 0.33
                p_away = row['away_win_prob'] if not pd.isna(row['away_win_prob']) else 0.33
                
                # 프로그레스 바 형태의 승률 표시
                cols = st.columns(3)
                cols[0].metric("Home", f"{p_home*100:.1f}%")
                cols[1].metric("Draw", f"{p_draw*100:.1f}%")
                cols[2].metric("Away", f"{p_away*100:.1f}%")
                
                # 배당률 정보
                st.caption(f"Market Odds: H:{row['home_win_odds']} | D:{row['draw_odds']} | A:{row['away_win_odds']}")

            with col3:
                st.markdown("#### 💰 Value Betting")
                edge = row['value_bet_edge']
                side = row['value_bet_side']
                
                if side != "No Bet" and not pd.isna(side):
                    st.markdown(f"""
                        <div class="value-bet">
                            🔥 {side} Recommendation<br/>
                            Edge: {edge*100:+.1f}%
                        </div>
                    """, unsafe_allow_html=True)
                    st.button(f"Share on KakaoTalk", key=f"share_{row['fixture_id']}")
                else:
                    st.markdown('<div class="no-bet">Negative Edge: No Value Found</div>', unsafe_allow_html=True)

            st.divider()

# 리프레시 로직
if auto_refresh:
    time.sleep(60)
    st.rerun()
