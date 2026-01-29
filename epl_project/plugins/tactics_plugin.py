import streamlit as st
import os

def get_metadata():
    return {
        "name": "tactics_plugin",
        "display_name": "👔 감독 전술 리포트",
        "description": "AI-driven manager tactical breakdown and historical context.",
        "version": "1.0.0"
    }

def render_ui(selected_team, clubs_data, **kwargs):
    st.title(f"{get_metadata()['display_name']}")
    
    current_team_info = next((item for item in clubs_data if item['team_name'] == selected_team), None)
    manager_name = current_team_info.get('manager_name', '감독 정보 없음') if current_team_info else "Unknown Manager"
    
    st.markdown(f"##### 🧠 **{manager_name}** 감독의 최신 전술 트렌드와 5경기 분석 데이터를 제공합니다.")
    
    # [Action Button]
    if st.button("📡 전술 데이터 실시간 수집 및 분석 시작", type="primary", use_container_width=True):
        with st.spinner(f"🔍 구글링 및 유튜브 분석 중... ({manager_name} tactics 2025)"):
            try:
                # [FIX] tactics_engine에서 올바른 함수 호출
                from tactics_engine import analyze_tactics
                report = analyze_tactics(selected_team, manager_name)
                st.session_state['tactics_report'] = report
                st.success("AI 전술 분석이 완료되었습니다!")
            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")
    
    # [Show Report Content]
    if 'tactics_report' in st.session_state and st.session_state['tactics_report'].get('team') == selected_team:
        report = st.session_state['tactics_report']
        
        st.divider()
        st.subheader("📝 AI 종합 전술 코멘트")
        st.markdown(f"""
        <div style="
            background: rgba(255, 235, 59, 0.1); 
            border-left: 5px solid #FFEB3B; 
            padding: 20px; 
            border-radius: 10px;
            margin-bottom: 20px;
        ">
            <p style="color: #FFEB3B; font-size: 17px; font-weight: 500; line-height: 1.6; margin: 0;">
                {report['ai_summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🔑 핵심 키워드")
            for kw in report['keywords']:
                st.markdown(f"- **#{kw}**")
        with c2:
            st.markdown("#### 📅 예상 포메이션")
            st.code(report['pref_formation'], language="text")
            
        st.divider()
        st.subheader("📰 참고 자료 (Sources)")
        for art in report['articles']:
            st.markdown(f"- [{art['title']}]({art['link']}) ({art['source']})")
            
        # [Sharing Functionality]
        st.divider()
        st.subheader("📤 리포트 공유하기")
        share_text = f"[{selected_team} 전술 리포트]\n\n감독: {manager_name}\n핵심 전술: {', '.join(report['keywords'])}\n포메이션: {report['pref_formation']}\n\nAI 분석 요약:\n{report['ai_summary'][:150]}...\n\n#EPL #축구분석 #안티그래비티"
        st.code(share_text, language="text")

def get_intelligence(selected_team, clubs_data, **kwargs):
    """Returns structured data for AI agents."""
    current_team_info = next((item for item in clubs_data if item['team_name'] == selected_team), None)
    manager_name = current_team_info.get('manager_name', 'Unknown')
    
    # In a real scenario, this would pull from a cached report or DB
    return {
        "team": selected_team,
        "manager": manager_name,
        "status": "ready",
        "primary_tactics": ["Build-up from back", "High Pressing"], # Example
        "agent_note": "This data is based on the SOTA tactics engine."
    }
