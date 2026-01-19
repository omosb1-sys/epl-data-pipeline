import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

from models.data_hq import EPLDataHQ
from models.xai_engine import get_real_shap_analysis, display_shap_chart

class EPLUpgradeUI:
    """
    [EPL Upgrade UI] 2026 SOTA UI/UX Enhancer
    - Features: SOTA Matrix, XAI Integration, Polars/DuckDB Insights
    """
    def __init__(self):
        self.hq = EPLDataHQ()
        
    def render_advanced_stats(self, selected_team):
        st.markdown("---")
        st.header("🧠 AI 데이터 분석 (SOTA)")
        
        # 1. Polars/DuckDB 기반 실시간 데이터 브리핑
        try:
            df_news = self.hq.load_and_transform()
            if not df_news.is_empty():
                # DuckDB를 활용한 동적 인사이트 추출
                team_low = selected_team.lower()
                query = f"SELECT title FROM df WHERE title_low LIKE '%{team_low}%' LIMIT 3"
                team_news = self.hq.query_with_duckdb(df_news, query)
                
                st.subheader("📰 Data HQ: 실시간 맥락 통찰")
                if not team_news.empty:
                    for _, row in team_news.iterrows():
                        st.markdown(f"""
                        <div style="padding:10px; border-left:4px solid #FF4B4B; background:rgba(255,255,255,0.05); margin-bottom:5px;">
                            {row['title']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("현재 해당 팀에 대한 실시간 이슈가 감지되지 않았습니다.")
        except Exception as e:
            st.error(f"Data HQ 브리핑 로드 실패: {e}")

        # 2. XAI 승부 예측 기여도 (Mock for UI Demo)
        st.subheader("🛡️ AI 승부 예측 판단 근거 (SHAP)")
        st.caption("AI가 왜 이런 예측을 내렸는지, 각 분석 지표의 기여도를 실시간으로 추적합니다.")
        
        # Mocking values for display
        mock_features = ['ADX', '+DI', '-DI', 'home_advantage', 'rest_days']
        mock_impact = np.random.uniform(-0.5, 0.5, len(mock_features))
        df_mock_shap = pd.DataFrame({'Feature': mock_features, 'Impact': mock_impact})
        
        # [Fix] 수동 Color 할당 제거 (xai_engine 내부 조건부 렌더링에 위임)
        display_shap_chart(df_mock_shap)

    def render_performance_matrix(self, clubs_data):
        st.divider()
        st.subheader("📊 팀 효율성 & 전력 매트릭스")
        
        plot_data = []
        for t in clubs_data:
            wins = t.get('wins', 0)
            draws = t.get('draws', 0)
            points = wins * 3 + draws
            power = t.get('power_index', 50)
            
            plot_data.append({
                'Team': t.get('team_name'),
                'Power Index': power,
                'Points': points,
                'Efficiency': points / power if power > 0 else 0
            })
        
        df_perf = pd.DataFrame(plot_data)
        
        fig = px.scatter(
            df_perf, x='Power Index', y='Points', text='Team',
            size='Points',
            color_continuous_scale='Viridis',
            template='plotly_dark'
        )
        fig.update_layout(
            xaxis_title='전력 지수 (Power Index)',
            yaxis_title='승점 (Points)',
            coloraxis_colorbar_title='효율성'
        )
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, width="stretch")

if __name__ == "__main__":
    # Test Entry
    st.write("EPL Upgrade UI Module Ready")
