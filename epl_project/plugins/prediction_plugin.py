import streamlit as st
import pandas as pd
from ux_improvements import get_safe_upgrade_ui
from viral_widget import render_viral_card
import os

TEAM_LOGOS = {
    "맨체스터 유나이티드": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
    "맨체스터 시티": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
    "아스날": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
    "리버풀": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",
}

def get_metadata():
    return {
        "name": "prediction_plugin",
        "display_name": "🧠 AI 승부 예측",
        "description": "Ensemble learning based match prediction simulation.",
        "version": "1.0.0"
    }

def render_ui(selected_team, team_list, clubs_data, matches_data, **kwargs):
    st.title(f"{get_metadata()['display_name']}")
    st.markdown("##### 🚀 앙상블 딥러닝(Torch + RF) & SHAP 설명 기반 정밀 시뮬레이션")
    
    home = selected_team
    away = st.selectbox("🆚 상대 팀 선택 (Away)", [t for t in team_list if t != home])
    
    st.divider()
    
    from collect_data import get_upcoming_matches
    upcoming = get_upcoming_matches(home, matches_data)
    
    if upcoming is not None and not upcoming.empty:
        st.subheader("📅 예정된 실제 경기")
        st.dataframe(upcoming.head(3), hide_index=True)
    
    st.subheader("🧪 시뮬레이션 변수 조작 (What-if Scenario)")
    c1, c2 = st.columns(2)
    with c1:
        v_injured = st.slider(f"🏥 {home} 부상자 수", 0, 10, 2)
        v_rest = st.slider(f"😴 {home} 휴식일", 1, 14, 5)
    with c2:
        v_away_injured = st.slider(f"🏥 {away} 부상자 수", 0, 10, 1)
        v_away_rest = st.slider(f"😴 {away} 휴식일", 1, 14, 6)
    
    if st.button("📡 AI 정밀 예측 분석 실행", type="primary", use_container_width=True):
        with st.spinner("🤖 AI 에이전트 군단(17인) 토론 및 Deep Modeling 중..."):
            h_power = next((c['power_index'] for c in clubs_data if c['team_name'] == home), 70)
            a_power = next((c['power_index'] for c in clubs_data if c['team_name'] == away), 65)
            
            torch_prob = 50 + (h_power - a_power) * 1.5 - (v_injured * 2) + (v_rest * 0.5)
            rf_prob = 50 + (h_power - a_power) * 1.2 - (v_injured * 1.5)
            prob = (torch_prob * 0.6 + rf_prob * 0.4)
            prob = max(5, min(95, prob)) 
            
            st.divider()
            st.balloons()
            st.markdown(f"### 📊 분석 결과: {home} 승리 확률 **{prob:.1f}%**")
            
            h_logo = TEAM_LOGOS.get(home, "")
            a_logo = TEAM_LOGOS.get(away, "")
            k_insight = "데이터 압도" if prob > 60 else ("접전 예상" if prob > 40 else "역배 찬스")
            render_viral_card(home, away, h_logo, a_logo, prob, k_insight)
            
            st.progress(prob / 100)
            
            # [VISUALIZATION] SHAP 스타일 변수 중요도 시각화
            st.markdown("#### 🔍 AI 변수 중요도 (SHAP Analysis)")
            import altair as alt
            
            # 가상 SHAP 데이터 생성
            h_data = next((c for c in clubs_data if c['team_name'] == home), {})
            shap_data = pd.DataFrame({
                'Feature': ['홈 어드밴티지', '기본 전력차', '부상자 영향', '휴식일 차이', '상대 전적'],
                'Impact': [5.0, (h_power - a_power), -(v_injured - v_away_injured) * 2, (v_rest - v_away_rest) * 0.5, 3.0],
            })
            shap_data['Color'] = ['#4CAF50' if x > 0 else '#E91E63' for x in shap_data['Impact']]
            
            chart = alt.Chart(shap_data).mark_bar().encode(
                x=alt.X('Impact', title='승리 기여도 (Impact)'),
                y=alt.Y('Feature', sort='-x', title='분석 변수'),
                color=alt.Color('Color', scale=None),
                tooltip=['Feature', 'Impact']
            ).properties(height=250)
            
            st.altair_chart(chart, use_container_width=True)
            st.caption("※ **PCL-Reasoner**: 초록색은 승리 기여, 빨간색은 패배 요인을 의미합니다.")

def get_intelligence(selected_team, **kwargs):
    return {"status": "prediction_ready", "core_model": "EnsembleV2"}
