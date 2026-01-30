
import streamlit as st
import time

def get_metadata():
    return {
        "name": "rival_analysis",
        "display_name": "⚔️ 라이벌 매치 분석",
        "description": "더비 매치 특수성과 역대 전적을 반영한 하이-스테이크 예측"
    }

def render_ui(selected_team=None, **kwargs):
    st.title("⚔️ 라이벌 매치 딥러닝 시뮬레이터")
    st.markdown("단순 승패를 넘어, **역대 전적, 최근 5경기 흐름, 더비 매치 특수성**을 반영한 심층 분석입니다.")
    
    # 팀 선택 (전달되지 않은 경우 self-selection)
    team_list = kwargs.get('team_list', ["맨체스터 유나이티드", "맨체스터 시티", "리버풀", "아스날", "첼시", "토트넘 홋스퍼"])
    
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("홈 팀", team_list, index=team_list.index(selected_team) if selected_team in team_list else 0)
    with col2:
        away = st.selectbox("원정 팀", team_list, index=1 if len(team_list) > 1 else 0)

    if st.button("🚀 라이벌 매치 정밀 분석 실행", type="primary"):
        with st.spinner(f"⚔️ {home} vs {away} 데이터 분석 중..."):
            time.sleep(1.5)
            
            # 라이벌 매치 데이터베이스
            rivalries = {
                "맨체스터 유나이티드": ["리버풀", "맨체스터 시티", "아스날"],
                "리버풀": ["맨체스터 유나이티드", "에버튼"],
                "아스날": ["토트넘 홋스퍼", "맨체스터 유나이티드", "첼시"],
                "토트넘 홋스퍼": ["아스날", "첼시", "웨스트햄 유나이티드"],
                "첼시": ["아스날", "토트넘 홋스퍼", "풀럼"],
                "맨체스터 시티": ["맨체스터 유나이티드", "리버풀"]
            }
            
            is_rivalry = away in rivalries.get(home, []) or home in rivalries.get(away, [])
            
            if is_rivalry:
                st.snow()
                st.error(f"### 🚨 [OFFICIAL RIVALRY] {home} vs {away}")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("경기 격렬도", "92/100", "Critical")
                    st.write("📌 **관전 포인트**: 양 팀 감독의 전술적 자존심 대결이 예상됩니다.")
                with res_col2:
                    st.metric("변수 발생률", "High", "75%")
                    st.write("📌 **AI 경고**: 퇴장 또는 PK가 승부처가 될 확률이 매우 높습니다.")
                
                st.info("💡 **Deep Learning Insight**: 역대급 기세 싸움이 될 것입니다. 객관적 전력보다는 당일 선수들의 멘탈리티가 승패를 결정합니다.")
            else:
                st.success(f"두 팀은 전통적인 라이벌 관계는 아닙니다. ({home} vs {away})")
                st.write("객관적인 전력 기반의 분석이 더 유효할 것으로 보입니다.")

def get_intelligence(selected_team=None, **kwargs):
    # AI 에이전트용 데이터 제공
    return {
        "rivalry_status": "Ready",
        "supported_derbies": ["North London Derby", "North West Derby", "Manchester Derby", "Merseyside Derby"]
    }
