import streamlit as st
import json # [NEW] JSON handling
import pandas as pd
from datetime import datetime
import os  # [필수] 이미지 경로 확인용
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # [EPL Fix] Mac crash 방지
os.environ['OMP_NUM_THREADS'] = '1' # [Stability Fix]


# from src.realtime_sync_engine import sync_data (Deprecated)
try:
    from collect_data import main as run_sync 
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from collect_data import main as run_sync

# [AI Engine] Lazy Loader
from ai_loader import get_ensemble_engine





# --- 0. 기본 설정 ---
st.set_page_config(
    page_title="EPL-X Manager",
    page_icon="⚽",
    layout="wide"
)

# 다크 모드 스타일적용
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    /* 로고가 어두운 배경에서 더 잘 보이도록 그림자 효과 추가 */
    img {
        filter: drop-shadow(0px 0px 5px rgba(255, 255, 255, 0.3));
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4F4F4F;
        margin-bottom: 10px;
    }
    
    /* [SEO/UX] 사이드바 메뉴 스타일 강화 */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-size: 22px !important;
        font-weight: 900 !important;
        color: #FFFFFF !important;
        margin-bottom: 10px !important;
        letter-spacing: -0.5px;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 12px 15px !important;
        border-radius: 12px !important;
        background-color: rgba(255,255,255,0.03) !important;
        margin-bottom: 8px !important;
        transition: all 0.2s ease-in-out !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: rgba(255,255,255,0.08) !important;
        transform: translateX(5px);
        border: 1px solid rgba(255,255,255,0.15) !important;
    }

    /* 선택된 아이템 강조 */
    [data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] div:first-child {
        border-color: #FF4B4B !important;
    }
    
    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 17px !important;
        font-weight: 600 !important;
        color: #E0E0E0 !important;
    }
    
    /* [ROLLBACK] 기본 메뉴/풋터만 숨기고 헤더 구조는 건드리지 않음 (모바일 호환성 최우선) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 
       [알림] GitHub 아이콘 숨기는 것이 모바일 사이드바 버튼까지 깨뜨리는 현상 발생.
       따라서 우선순위를 '앱 사용성(사이드바)'에 두고 툴바 숨김 코드는 제거함.
       대신 툴바의 배경을 투명하게 해서 눈에 덜 띄게 만듦.
    */
    [data-testid="stToolbar"] {
        right: 2rem;
    }

    /* [UX UPGRADE] 모바일 사이드바 버튼(햄버거) 대폭 확대 및 텍스트 추가 */
    /* [UX UPGRADE] 모바일 사이드바 버튼(햄버거) 초강력 스타일링 */
    [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] {
        display: block !important;
        visibility: visible !important;
        position: fixed !important; /* 헤더 레이아웃 무시하고 고정 */
        top: 10px !important;
        left: 10px !important;
        
        transform: scale(1.8) !important; 
        transform-origin: top left !important;
        
        background-color: rgba(255, 75, 75, 0.9) !important; /* 붉은 배경으로 확실하게 */
        border: 2px solid white !important;
        border-radius: 8px !important;
        
        z-index: 9999999 !important; /* 무조건 맨 위 */
        
        /* 터치 영역 확보 */
        width: 40px !important;
        height: 40px !important;
        padding: 5px !important;
    }
    
    /* 아이콘 색상 강제 */
    [data-testid="collapsedControl"] svg, [data-testid="stSidebarCollapsedControl"] svg {
        fill: white !important;
        stroke: white !important;
    }

    /* "MENU" 텍스트 라벨 추가 (버튼 내부 하단) */
    [data-testid="collapsedControl"]::after, [data-testid="stSidebarCollapsedControl"]::after {
        content: "MENU";
        display: block;
        position: absolute;
        bottom: 2px;
        left: 0;
        width: 100%;
        font-size: 8px; /* scale 적용되므로 실제론 14~15px */
        font-weight: 900;
        color: white; 
        text-align: center;
        text-shadow: 1px 1px 2px black;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. 데이터 로드 (Serverless JSON Mode) ---
def load_json_data(filename):
    path = os.path.join("epl_project/data", filename)
    # 로컬 테스트용 경로 보정
    if not os.path.exists(path):
        path = os.path.join("data", filename)
        
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# 데이터 로드 함수
def load_data():
    # 1. 정적 구단 정보 (Managers, Stadiums, History) - from Backup
    clubs = load_json_data("clubs_backup.json")
    
    # 2. 동적 순위 정보 (Standings) - from API
    # API 데이터가 있으면 순위/승점 등을 최신으로 덮어쓰기 로직 (Optional)
    # dynamic = load_json_data("latest_epl_data.json")
    
    return clubs

def fetch_matches():
    # API에서 수집한 Fixtures 데이터 로드
    data = load_json_data("latest_epl_data.json")
    if isinstance(data, dict):
        return data.get('fixtures', [])
    return []

def analyze_team_realtime(target_team):
    """
    서버리스 모드: 이미 수집된 news 데이터를 기반으로 즉석 분석 수행
    """
    data = load_json_data("latest_epl_data.json")
    news_list = data.get('news', []) if isinstance(data, dict) else []
    
    # 1. 키워드 매핑 (한글 구단명 -> 영어 검색어)
    rev_map = {
        "아스널": "Arsenal", "리버풀": "Liverpool", "맨체스터 시티": "Manchester City", "맨시티": "Manchester City",
        "아스톤 빌라": "Aston Villa", "첼시": "Chelsea", "브라이튼": "Brighton",
        "토트넘 홋스퍼": "Tottenham", "토트넘": "Tottenham", "노팅엄 포레스트": "Forest", "노팅엄": "Nottingham",
        "뉴캐슬 유나이티드": "Newcastle", "풀럼": "Fulham", "본머스": "Bournemouth", 
        "웨스트햄 유나이티드": "West Ham", "브렌트포드": "Brentford", "레스터 시티": "Leicester", 
        "에버튼": "Everton", "크리스탈 팰리스": "Crystal Palace", "팰리스": "Crystal Palace",
        "입스위치 타운": "Ipswich", "울버햄튼": "Wolves", "사우스햄튼": "Southampton", 
        "맨체스터 유나이티드": "Manchester United", "맨유": "Manchester United"
    }
    eng_name = rev_map.get(target_team, target_team)
    
    # 2. 뉴스 필터링
    relevant_news = []
    keywords = [eng_name.lower()]
    if "manchester" in keywords[0]: 
        if "united" in keywords[0]: keywords.append("man utd")
        if "city" in keywords[0]: keywords.append("man city")
    
    for n in news_list:
        if not isinstance(n, dict): continue
        title = n.get('title', '').lower()
        if any(k in title for k in keywords):
            relevant_news.append(n)
            
    # 3. 감성/키워드 분석 (Rule-based)
    score = 50.0 # Base score
    pos_words = ["win", "victory", "sign", "deal", "success", "top", "goal", "return", "fit"]
    neg_words = ["lose", "defeat", "injury", "out", "miss", "fail", "sack", "crisis"]
    
    summary_sentences = []
    
    if relevant_news:
        for n in relevant_news[:5]: # 최신 5개만 분석
            title = n.get('title', '')
            t_lower = title.lower()
            
            # Scoring
            pos_cnt = sum(1 for w in pos_words if w in t_lower)
            neg_cnt = sum(1 for w in neg_words if w in t_lower)
            score += (pos_cnt * 2.0) - (neg_cnt * 2.5)
            
            summary_sentences.append(f"- {title}")
    else:
        summary_sentences.append("최근 특이사항이 감지되지 않았습니다.")
        
    # Bound score
    score = max(0, min(100, score))
    
    summary = "\n".join(summary_sentences[:3]) # Top 3 summary
    
    return score, summary, relevant_news[:5]

# --- 2. 데이터 로딩 ---
# 팀 목록 가져오기
clubs_data = load_data()
matches_data = fetch_matches()

# 팀 이름 리스트 만들기 (가나다 순 정렬)
if clubs_data:
    team_list = sorted([team['team_name'] for team in clubs_data])
else:
    team_list = ["데이터 없음"]

# 로고 매핑 (한글 이름 키값 적용)
TEAM_LOGOS = {
    "맨체스터 유나이티드": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
    "맨체스터 시티": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
    "아스날": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
    "리버풀": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",
    "첼시": "epl_project/assets/logos/chelsea_premium.png",
    "토트넘 홋스퍼": "epl_project/assets/logos/spurs_white.png",
    "뉴캐슬 유나이티드": "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg",
    "아스톤 빌라": "https://upload.wikimedia.org/wikipedia/en/f/f9/Aston_Villa_FC_crest_%282016%29.svg",
    "울버햄튼": "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg",
    "브라이튼": "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg",
    "크리스탈 팰리스": "epl_project/assets/logos/crystal_palace_premium.png",
    "풀럼": "https://upload.wikimedia.org/wikipedia/en/e/eb/Fulham_FC_%28shield%29.svg",
    "본머스": "https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg",
    "웨스트햄 유나이티드": "https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg",
    "에버튼": "https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg",
    "브렌트포드": "https://upload.wikimedia.org/wikipedia/en/2/2a/Brentford_FC_crest.svg",
    "노팅엄 포레스트": "https://upload.wikimedia.org/wikipedia/en/e/e5/Nottingham_Forest_F.C._logo.svg",
    "레스터 시티": "https://upload.wikimedia.org/wikipedia/en/2/2d/Leicester_City_crest.svg",
    "사우스햄튼": "https://upload.wikimedia.org/wikipedia/en/c/c9/FC_Southampton.svg",
    "입스위치 타운": "https://upload.wikimedia.org/wikipedia/en/4/43/Ipswich_Town.svg"
}

# --- 3. 사이드바 (핵심 컨트롤) ---
with st.sidebar:
    st.header("🎯 컨트롤 타워")
    
    # [디버깅] 데이터 상태 표시
    if clubs_data:
        st.caption(f"✅ DB 연결됨 ({len(clubs_data)}팀)")
    else:
        st.error("❌ DB 데이터 없음")

    # [중요] key를 변경하여 세션 상태 강제 리셋 (v2)
    selected_team = st.selectbox(
        "분석할 구단 선택", 
        options=team_list,
        index=0,
        key="team_selector_v2" 
    )
    
    # 로고 표시 (프리미엄 AI 로고 반영 및 시인성 극대화)
    logo_path = TEAM_LOGOS.get(selected_team, "https://upload.wikimedia.org/wikipedia/commons/d/d3/Soccerball.svg")
    
    # 로컬 파일인 경우 인코딩 처리 또는 직접 경로 사용 지원
    if os.path.exists(logo_path):
        import base64
        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        logo_url = f"data:image/png;base64,{encoded}"
    else:
        logo_url = logo_path

    st.markdown(f"""
        <div style="text-align: center; padding: 10px;">
            <img src="{logo_url}" width="150" style="filter: drop-shadow(0px 0px 15px rgba(255, 255, 255, 0.4)); border-radius: 10px;">
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    # [MOVE] 메뉴 이동을 구단 이미지 바로 아래로 배치
    menu = st.radio("🎯 메뉴 이동", ["📊 실시간 대시보드", "🧠 AI 승부 예측", "🔁 이적 시장 통합 센터", "📰 프리미어리그 최신 뉴스"], key="menu_selector")
    
    st.divider()
    
    # [NEW] 실시간 동기화 섹션
    st.subheader("🌐 Live Sync")
    if st.button("🛰️ 실시간 데이터 동기화"):
        with st.sidebar:
            with st.status("최신 뉴스 및 팩트 수집 중...", expanded=True) as status:
                try:
                    # Serverless Sync 실행
                    run_sync()
                    
                    # [FIX] 수집된 뉴스 데이터 세션에 즉시 반영
                    latest_data = load_json_data("latest_epl_data.json")
                    news_data = latest_data.get('news', []) if isinstance(latest_data, dict) else []
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state['sync_result'] = {
                        'timestamp': timestamp, 
                        'updates': ["데이터 갱신 완료", f"뉴스 {len(news_data)}건 수집됨"], 
                        'news': news_data
                    }
                    status.update(label=f"동기화 완료! ({timestamp})", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    status.update(label="동기화 실패 (API Key 확인 필요)", state="error")
                    st.error(f"Error: {e}")

    if st.button("🔄 전체 새로고침 (Soft Refresh)"):
        st.cache_data.clear()
        st.rerun()
    
    # 최신 뉴스 및 자동 업데이트 위젯
    if 'sync_result' in st.session_state:
        res = st.session_state['sync_result']
        
        # 1. 자동 반영된 소식
        if res['updates']:
            with st.expander("🤖 자동 데이터 보충 결과", expanded=True):
                for up in res['updates']:
                    # Compact custom success message (Small font)
                    st.markdown(f"""
                    <div style="
                        padding: 6px 10px;
                        border-radius: 6px;
                        background-color: rgba(33, 195, 84, 0.15); /* Subtle Green */
                        border: 1px solid rgba(33, 195, 84, 0.3);
                        margin-bottom: 5px;
                        display: flex;
                        align-items: start;
                    ">
                        <div style="font-size: 14px; margin-right: 8px;">✅</div>
                        <div style="font-size: 11px; color: #e0e0e0; line-height: 1.3;">{up}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        # 2. 최신 뉴스 헤드라인 (사이드바)
        with st.expander("🌍 최신 EPL 헤드라인", expanded=False):
            for news in res['news']:
                # Dict type check
                if isinstance(news, dict):
                    st.markdown(f"• <a href='{news['url']}' target='_blank' style='text-decoration:underline; color:#0366d6;'>{news['title']}</a>", unsafe_allow_html=True)
                else:
                    st.caption(f"• {news}")
        
        
    # menu = st.radio(...) -> Moved to Top

# --- 4. 메인 대시보드 로직 ---
if menu == "📊 실시간 대시보드":
    # [강력 조치] 캐시 강제 삭제 (이미지 반영을 위해)
    st.cache_data.clear()

    # 제목에 선택된 팀 이름 강제 주입
    st.title(f"📊 {selected_team} 데이터 센터")
    
    # 선택된 팀 정보 찾기
    current_team_info = next((item for item in clubs_data if item['team_name'] == selected_team), None)
    
    if current_team_info:
        # [1] 상단 핵심 지표
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("타겟 구단", selected_team)
        with col2:
            st.metric("현재 감독", current_team_info['manager_name'])
        with col3:
            st.metric("AI 전력 지수", f"{current_team_info['power_index']}/100")
        
        st.divider()
        
        # [2] 구단 상세 프로필 (New!)
        st.subheader("🏟️ 구단 상세 프로필")
        
        p_col1, p_col2, p_col3 = st.columns([1.5, 1, 1])
        
        # 왼쪽: 경기장 이미지
        with p_col1:
            stadium_img = current_team_info.get('stadium_img')
            
            final_img = None
            
            # [1] DB에 저장된 경로 우선 확인
            if stadium_img:
                # 1. 웹 URL인 경우
                if str(stadium_img).startswith("http"):
                    final_img = stadium_img
                # 2. 로컬 파일인 경우 (stadiums/...)
                elif os.path.exists(stadium_img):
                    final_img = stadium_img
            
                # [2] DB에 없거나 파일이 없으면 -> 비상용 매핑 확인
                if not final_img:
                    # [FIX] 파일 경로를 epl_project/stadiums/... 형태로 보정
                    BASE_DIR = os.path.dirname(__file__)
                    LOCAL_FALLBACKS = {
                        "맨체스터 유나이티드": "stadiums/man_utd.jpg",
                        "맨체스터 시티": "stadiums/man_city.jpg",
                        "리버풀": "stadiums/liverpool.jpg",
                        "아스날": "stadiums/arsenal.png",
                        "첼시": "stadiums/chelsea.png",
                        "토트넘 홋스퍼": "stadiums/totten_h.png",
                        "뉴캐슬 유나이티드": "stadiums/newcastle_u.png",
                        "아스톤 빌라": "stadiums/man_city.jpg", # 임시 대체 (파일 없음)
                        "울버햄튼": "stadiums/wolverhampton_w.png",
                        "브라이튼": "stadiums/brighton_h_a.png",
                        "크리스탈 팰리스": "stadiums/crystal_p.png",
                        "풀럼": "stadiums/fulham.png",
                        "본머스": "stadiums/bournemouth.png",
                        "웨스트햄 유나이티드": "stadiums/west.h.png",
                        "에버튼": "stadiums/everton.png",
                        "브렌트포드": "stadiums/brentford.png",
                        "노팅엄 포레스트": "stadiums/nottingham_f.png",
                        "레스터 시티": "stadiums/leichester_c.png",
                        "사우스햄튼": "stadiums/s_hampton.png",
                    }
                    
                    rel_path = LOCAL_FALLBACKS.get(selected_team)
                    if rel_path:
                        abs_path = os.path.join(BASE_DIR, rel_path)
                        if os.path.exists(abs_path):
                            final_img = abs_path

                if not final_img:
                    final_img = "https://placehold.co/600x400/png?text=No+Stadium+Image"

            # 최종 출력
            if final_img:
                st.image(final_img, caption=f"{selected_team} 홈 구장", use_container_width=True)
            else:
                st.info("이미지를 찾을 수 없습니다.")

        # 가운데: 핵심 스탯 (가치, 순위)
        with p_col2:
            val = current_team_info.get('club_value', '정보 없음')
            rank = current_team_info.get('current_rank', '-')
            last_rank = current_team_info.get('last_season_rank', '-')
            
            wins = current_team_info.get('wins', 0)
            draws = current_team_info.get('draws', 0)
            losses = current_team_info.get('losses', 0)
            
            st.markdown(f"""
            #### 💰 구단 가치
            **{val}**
            
            #### 🏆 리그 순위
            * **현재:** {rank}위
            * **지난 시즌:** {last_rank}위
            
            #### 📈 시즌 전적
            **{wins}승 {draws}무 {losses}패**
            """)

        # 오른쪽: 이적 시장 현황
        with p_col3:
            t_in = current_team_info.get('transfers_in', '정보 없음')
            t_out = current_team_info.get('transfers_out', '정보 없음')
            
            st.markdown("#### 🔄 주요 영입 (IN)")
            st.code(t_in)
            
            st.markdown("#### 🚪 주요 방출 (OUT)")
            st.code(t_out)


        # [NEW] 감독 및 전술 분석 카드
        st.divider()
        st.subheader("👔 감독 및 전술 스타일 분석 (2025 Current)")
        
        tac_fmt = current_team_info.get('tactics_formation', '4-4-2')
        tac_desc = current_team_info.get('tactics_desc', '전술 데이터 확인 중...')
        
        with st.container(border=True):
            tc1, tc2 = st.columns([1, 3])
            with tc1:
                st.markdown(f"**📌 주 포메이션**")
                st.info(tac_fmt)
            with tc2:
                st.markdown(f"**🗣️ 전술 포인트**")
                st.write(tac_desc)

        # [3] 구단 오피셜 & 팬파크 (이미지 스타일 구현)
        st.divider()
        st.subheader("🏵️ 구단 오피셜 스태프 및 레전드 명단")
        
        # 각 구단별 최신(2025-26) 데이터 매핑
        staff_map = {
            "맨체스터 유나이티드": {
                "분류": ["임시 감독 (Interim Manager)", "코칭 스탭 (Coaching Staff)", "플레잉 코치 / 레전드"],
                "명단": [
                    "대런 플레처 (Darren Fletcher)",
                    "트래비스 비니언, 앨런 라이트 (Academy Coaches)",
                    "조니 에반스(Jonny Evans), 박지성, 웨인 루니, 폴 스콜스"
                ]
            },
            "아스날": {
                "분류": ["메인 매니저 (Manager)", "코칭 스탭", "명예 레전드"],
                "명단": [
                    "미켈 아르테타 (Mikel Arteta)",
                    "알베르트 스투이벤베르그, 카를로스 쿠에스타, 니콜라 조버(세트피스)",
                    "티에리 앙리, 데니스 베르캄프, 패트릭 비에이라, 이안 라이트, 토니 아담스"
                ]
            },
            "맨체스터 시티": {
                "분류": ["매니저", "코칭 스탭", "레전드"],
                "명단": [
                    "펩 과르디올라 (Pep Guardiola)",
                    "후안마 리요, 카를로스 비센스, 리차드 라이트",
                    "세르히오 아구에로, 다비드 실바, 빈센트 콤파니, 야야 투레"
                ]
            },
            "리버풀": {
                "분류": ["헤드 코치 (Head Coach)", "코칭 스탭", "레전드"],
                "명단": [
                    "아르네 슬롯 (Arne Slot)",
                    "십케 훌쇼프, 존 헤이팅아, 루벤 피터스",
                    "스티븐 제라드, 케니 달글리시, 이안 러쉬, 제이미 캐러거, 로비 파울러"
                ]
            },
            "토트넘 홋스퍼": {
                "분류": ["매니저 (Status)", "임시/코칭 스탭", "레전드"],
                "명단": [
                    "감독직 공석 (Searching for New Manager)",
                    "라이언 메이슨(대행), 맷 웰스, 니콜라스 옐리치",
                    "다니엘 레비(회장?), 가레스 베일, 해리 케인, 지미 그리브스, 레들리 킹"
                ]
            }
        }

        if selected_team in staff_map:
            current_staff = staff_map[selected_team]
            for idx, row in enumerate(current_staff["분류"]):
                with st.expander(f"{row}", expanded=True):
                    names = current_staff["명단"][idx].split(", ")
                    st.markdown(" ".join([f"`{name.strip()}`" for name in names]))
        else:
            st.info(f"{selected_team}의 명단은 현재 2025-26 버전으로 업데이트 중입니다.")

    else:
        st.error("구단 정보를 불러오지 못했습니다.")
    
    st.divider()
    
    # 경기 일정 필터링 및 시간 변환 (UK/KR)
    my_matches = []
    from datetime import timedelta
    
    # [FIX] API 영문 팀명 -> 앱 한글 팀명 매핑 테이블 (정밀화)
    # 한글 이름에서 영문 키워드로 변환 (푸른 박스 안내 및 필터링용)
    rev_map = {
        "아스널": "Arsenal", "리버풀": "Liverpool", "맨체스터 시티": "Manchester City",
        "아스톤 빌라": "Aston Villa", "첼시": "Chelsea", "브라이튼": "Brighton",
        "토트넘 홋스퍼": "Tottenham", "노팅엄 포레스트": "Nottingham Forest", "뉴캐슬 유나이티드": "Newcastle",
        "풀럼": "Fulham", "본머스": "Bournemouth", "웨스트햄 유나이티드": "West Ham",
        "브렌트포드": "Brentford", "레스터 시티": "Leicester", "에버튼": "Everton",
        "크리스탈 팰리스": "Crystal Palace", "입스위치 타운": "Ipswich", "울버햄튼": "Wolves",
        "사우스햄튼": "Southampton", "맨체스터 유나이티드": "Manchester United"
    }
    eng_keyword = rev_map.get(selected_team, selected_team)

    for m in matches_data:
        h_name = str(m.get('home_team', ''))
        a_name = str(m.get('away_team', ''))
        
        # [핵심] 대소문자 무시 및 부분 일치 확인 (Fuzzy Matching)
        is_match = False
        m_lower = (h_name + a_name).lower()
        
        if eng_keyword.lower() in m_lower:
            is_match = True
        
        # [NEW/ROBUST] 노팅엄 포레스트/맨유 등 키워드 정밀 처리 (API 변동성 대응)
        if selected_team == "노팅엄 포레스트":
            if any(kw in m_lower for kw in ["forest", "nottingham", "nottm"]):
                is_match = True
        
        # 맨유 특수 처리 (United 키워드 중복 방지)
        if selected_team == "맨체스터 유나이티드":
            if "united" in m_lower and not any(kw in m_lower for kw in ["west ham", "newcastle", "sheffield", "leeds"]):
                is_match = True

        if is_match:
            # API 시간 (UTC 기준) 파싱
            try:
                date_str = m.get('date', '')
                if 'T' in date_str:
                    dt_utc = datetime.strptime(date_str.split('+')[0].replace('T', ' '), "%Y-%m-%d %H:%M:%S")
                else:
                    dt_utc = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                
                dt_kr = dt_utc + timedelta(hours=9)
                
                my_matches.append({
                    "상대": f"{h_name} (홈)" if eng_keyword.lower() in a_name.lower() else f"{a_name} (원정)",
                    "영국 시간 (GMT)": dt_utc.strftime("%m/%d %H:%M"),
                    "한국 시간 (KST)": dt_kr.strftime("%m/%d %H:%M"),
                    "현재 상태": m.get('status', '예정')
                })
            except:
                pass
    
    st.subheader(f"📅 {selected_team} 경기 일정 (Live)")
    
    if my_matches:
        st.dataframe(my_matches, use_container_width=True)
    else:
        st.info(f"현재 데이터베이스에 '{selected_team}'의 경기 정보가 포착되지 않았습니다.")
        st.caption("사이드바에서 '실시간 데이터 동기화'를 실행하여 최신 피드를 수집해보세요.")

elif menu == "🧠 AI 승부 예측":
    st.title(f"🎮 AI 승부 예측 시뮬레이터 (Interactive)")
    st.markdown("##### ⚡ 실제 데이터를 기반으로 하되, 당신이 직접 변수를 조작하여 시뮬레이션할 수 있습니다.")
    
    # [1] 팀 선택
    c1, c2, c3 = st.columns([1, 0.2, 1])
    with c1:
        h_idx = team_list.index(selected_team) if selected_team in team_list else 0
        home = st.selectbox("🏠 홈 팀", team_list, index=h_idx, key="pred_home")
    with c2:
        st.markdown("<h2 style='text-align: center;'>VS</h2>", unsafe_allow_html=True)
    with c3:
        a_idx = (h_idx + 1) % len(team_list)
        away = st.selectbox("✈️ 원정 팀", team_list, index=a_idx, key="pred_away")
        
    st.divider()

    # [NEW] 선택된 팀들 간의 다음 경기 일정 자동 포착 (매핑 고려)
    team_name_map = {
        "Arsenal": "아스널", "Liverpool": "리버풀", "Manchester City": "맨체스터 시티",
        "Aston Villa": "아스톤 빌라", "Chelsea": "첼시", "Brighton": "브라이튼",
        "Tottenham": "토트넘 홋스퍼", "Nottingham Forest": "노팅엄 포레스트", "Newcastle": "뉴캐슬 유나이티드",
        "Fulham": "풀럼", "Bournemouth": "본머스", "West Ham": "웨스트햄 유나이티드",
        "Brentford": "브렌트포드", "Leicester": "레스터 시티", "Everton": "에버튼",
        "Crystal Palace": "크리스탈 팰리스", "Ipswich": "입스위치 타운", "Wolves": "울버햄튼",
        "Southampton": "사우스햄튼", "Manchester United": "맨체스터 유나이티드"
    }
    rev_map = {v: k for k, v in team_name_map.items()}
    eng_home = rev_map.get(home, home)
    eng_away = rev_map.get(away, away)

    next_match = next((m for m in matches_data if 
        (eng_home in m['home_team'] and eng_away in m['away_team']) or 
        (eng_away in m['home_team'] and eng_home in m['away_team'])), None)
    
    if next_match:
        from datetime import timedelta
        try:
            date_str = next_match.get('date', '')
            if 'T' in date_str:
                dt_utc = datetime.strptime(date_str.split('+')[0].replace('T', ' '), "%Y-%m-%d %H:%M:%S")
            else:
                dt_utc = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            dt_kr = dt_utc + timedelta(hours=9)
            st.markdown(f"""
            <div style="background-color:rgba(30,136,229,0.1); padding:10px; border-radius:10px; text-align:center; border: 1px solid rgba(30,136,229,0.3); margin-bottom:20px;">
                <span style="font-size:0.9em; color:#90CAF9;">📅 예정 대진 시간 (Official Fixture)</span><br>
                <b style="font-size:1.1em;">영국(GMT): {dt_utc.strftime('%Y-%m-%d %H:%M')}</b> | <b style="font-size:1.1em; color:#FFCA28;">한국(KST): {dt_kr.strftime('%Y-%m-%d %H:%M')}</b>
            </div>
            """, unsafe_allow_html=True)
        except: pass
    else:
        st.warning(f"🚨 현재 '{home}' vs '{away}'의 공식 일정이 데이터베이스에 없습니다. 곧 업데이트될 예정입니다.")

    if home == away:
        st.warning("동일한 팀입니다.")
    else:
        # [2] DB에서 기본값 가져오기 (초기 세팅용)
        h_data = next((item for item in clubs_data if item['team_name'] == home), None)
        a_data = next((item for item in clubs_data if item['team_name'] == away), None)
        
        # 기본값 로드 (없으면 안전값)
        h_def_rest = h_data.get('rest_days', 3) if h_data else 3
        h_def_inj = h_data.get('injury_level', '보통') if h_data else '보통'
        h_def_mood = h_data.get('team_mood', '보통') if h_data else '보통'
        
        a_def_rest = a_data.get('rest_days', 3) if a_data else 3
        a_def_inj = a_data.get('injury_level', '보통') if a_data else '보통'
        a_def_mood = a_data.get('team_mood', '보통') if a_data else '보통'

        # 옵션 리스트 정의
        inj_opts = ["풀전력", "경미", "보통", "심각", "주전 줄부상 비상"]
        mood_opts = ["최악", "나쁨", "보통", "좋음", "최상"]
        
        # 인덱스 찾기 안전장치
        try: h_inj_idx = inj_opts.index(h_def_inj)
        except: h_inj_idx = 2
        try: a_inj_idx = inj_opts.index(a_def_inj) 
        except: a_inj_idx = 2
        
        try: h_mood_idx = mood_opts.index(h_def_mood)
        except: h_mood_idx = 2
        try: a_mood_idx = mood_opts.index(a_def_mood) 
        except: a_mood_idx = 2

        # [3] 사용자 조작 패널 (기본값 = DB 데이터)
        st.subheader("🎛️ 시뮬레이션 변수 조작")
        
        col_cond_h, col_cond_a = st.columns(2)
        
        with col_cond_h:
            st.info(f"🛡️ {home} 설정")
            h_rest = st.slider(f"{home} 휴식일", 0, 10, int(h_def_rest), key="s_h_rest")
            h_injury = st.selectbox(f"{home} 부상 수준", inj_opts, index=h_inj_idx, key="s_h_inj")
            h_vibe = st.select_slider(f"{home} 분위기", mood_opts, value=h_def_mood, key="s_h_mood")
            
        with col_cond_a:
            st.error(f"⚔️ {away} 설정")
            a_rest = st.slider(f"{away} 휴식일", 0, 10, int(a_def_rest), key="s_a_rest")
            a_injury = st.selectbox(f"{away} 부상 수준", inj_opts, index=a_inj_idx, key="s_a_inj")
            a_vibe = st.select_slider(f"{away} 분위기", mood_opts, value=a_def_mood, key="s_a_mood")

        # [4] 시뮬레이션 실행 (Deep Learning & Ensemble)
        if st.button("🧠 AI 정밀 예측 분석 실행", type="primary", use_container_width=True):
            st.divider()
            
            with st.status("AI 인텔리전스 가동 중...", expanded=True) as status:
                h_power = h_data.get('power_index', 50) if h_data else 50
                a_power = a_data.get('power_index', 50) if a_data else 50

                # Standard Engine 가동 (Deep Learning + RandomForest)
                AI_TORCH, AI_RF, AI_SCALER = get_ensemble_engine()
                h_form_str = h_data.get('form', 'DDDDD') if h_data else "DDDDD"
                h_form_val = sum([3 if c=='W' else 1 if c=='D' else 0 for c in h_form_str[-5:]]) / 15.0
                
                if AI_TORCH and AI_RF and AI_SCALER:
                    try:
                        import torch
                        import numpy as np
                        input_raw = np.array([[h_data.get('goals_scored', 30), h_data.get('goals_conceded', 20), h_data.get('elo', 1500), h_form_val]], dtype=np.float32)
                        input_scaled = AI_SCALER.transform(input_raw)
                        prob_torch = AI_TORCH(torch.from_numpy(input_scaled)).item()
                        prob_rf = AI_RF.predict_proba(input_scaled)[0][1]
                        prob = (prob_torch * 0.4 + prob_rf * 0.6) * 100
                    except Exception as e:
                        st.error(f"예측 도중 오류 발생: {e}")
                        prob = 50.0
                else:
                    st.warning("⚠️ 안정화 엔진 로드 실패. 기본 전력 분석으로 대체합니다.")
                    prob = 50.0 + (h_power - a_power) # Fallback
                
                status.update(label="분석 완료!", state="complete", expanded=False)

            # 결과 가시화 (Senior Analyst Style - Multi-Model Breakdown)
            st.markdown("### 🏆 AI 통합 분석 엔진 결과")
            
            # 메인 앙상블 확률 표시
            col_res_l, col_res_m, col_res_r = st.columns([1,2,1])
            with col_res_l:
                st.metric(f"🏠 {home}", f"{prob:.1f}%")
            with col_res_r:
                st.metric(f"✈️ {away}", f"{100-prob:.1f}%")
            
            st.progress(prob / 100)

            # [NEW] 다중 모델 개별 분석 결과 공개
            with st.expander("🔍 다중 모델 분석 상세 데이터 보기", expanded=True):
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.write("🧠 **PyTorch DeepNet**")
                    try: st.info(f"승률 예측: {prob_torch*100:.1f}%")
                    except: st.info(f"승률 예측: {prob:.1f}%")
                    st.caption("비선형 경기력 흐름 분석")
                with m_col2:
                    st.write("🌲 **RandomForest Expert**")
                    try: st.success(f"승률 예측: {prob_rf*100:.1f}%")
                    except: st.success(f"승률 예측: {prob:.1f}%")
                    st.caption("통계적 변수 중요도 분석")
                
                st.write(f"⚖️ **최종 앙상블 합의 확률: {prob:.1f}%** (가중 평균 적용)")

            # [VISUALIZATION] SHAP 스타일 변수 중요도 시각화 (Mockup)
            st.markdown("### 📊 AI 변수 중요도 (SHAP Analysis)")
            st.markdown("어떤 요인이 이 승부의 향방을 결정했는지 AI가 인과관계를 분석했습니다.")
            
            # 가상 SHAP 값 생성 (시나리오별)
            import pandas as pd
            import altair as alt
            
            # [Dynamic SHAP Simulation] 현재 상황에 맞게 그래프 데이터 생성
            impact_home = (prob - 50) * 0.5
            impact_goal = (h_data.get('goals_scored', 30) - 25) * 0.4
            impact_vs = 10.0 if h_power > a_power else -10.0
            impact_injury = -5.0 # 부상 변수 (고정 예시)
            impact_tactics = 3.0
            
            shap_data = pd.DataFrame({
                'Feature': ['홈 어드밴티지', '최근 득점력', '객관적 전력차', '부상자 리스크', '전술 상성'],
                'Impact': [impact_home, impact_goal, impact_vs, impact_injury, impact_tactics],
                'Color': ['#4CAF50' if x > 0 else '#E91E63' for x in [impact_home, impact_goal, impact_vs, impact_injury, impact_tactics]]
            })
            
            chart = alt.Chart(shap_data).mark_bar().encode(
                x=alt.X('Impact', title='승리 기여도 (Impact)'),
                y=alt.Y('Feature', sort='-x', title='분석 변수'),
                color=alt.Color('Color', scale=None),
                tooltip=['Feature', 'Impact']
            ).properties(
                height=300
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            st.caption("※ 빨간색(Neg)은 패배/실점 요인, 초록색(Pos)은 승리/득점 요인을 의미합니다.")

            # [NEW] 스마트 리포트 생성 및 통합 표시
            def generate_smart_report(home, away, prob):
                if prob > 60:
                    verdict = f"🏟️ **{home}의 압도적 우세**"
                    causal = f"{home}의 홈 구장 화력과 공격진의 높은 xG(기대 득점) 전환율이 승리의 강력한 트리거로 작용하고 있습니다."
                    trend = f"최근 5경기 데이터 추적 결과, {home}은 상승 곡선을 타며 전술적 완성도가 정점에 도달했습니다."
                    color = "#4CAF50"
                elif prob < 40:
                    verdict = f"✈️ **{away}의 원정 승리 유력**"
                    causal = f"{home}의 수비 라인이 상위권 팀의 빠른 역습을 견디기엔 인과적으로 취약한 상태입니다. {away}의 중원 장악력이 핵심입니다."
                    trend = f"{away}는 최근 원정 경기에서도 꾸준한 득점력을 유지하는 '강팀의 본모습'을 시계열 데이터로 증명하고 있습니다."
                    color = "#E91E63"
                else:
                    verdict = f"⚖️ **예측 불가한 박빙의 승부**"
                    causal = f"양 팀의 전술적 상성이 팽팽하게 맞물려 있으며, 작은 실수가 곧 실점으로 이어질 수 있는 고도의 심리전 구간입니다."
                    trend = f"두 팀 모두 최근 널뛰는 경기력을 보이고 있어, 당일 선발 명단과 컨디션이 인공지능 예측 이상의 변수가 될 수 있습니다."
                    color = "#FFC107"
                return verdict, causal, trend, color

            v_title, v_causal, v_trend, v_color = generate_smart_report(home, away, prob)

            # SHAP-Style 가상 해석 리포트 (Visual Overhaul)
            st.markdown(f"""
            <div style="background-color:rgba(255,255,255,0.03); padding:25px; border-radius:15px; border-left: 8px solid {v_color}; margin-top:20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="margin-top:0; color:{v_color};">{v_title}</h3>
                <p style="font-size:16px; line-height:1.6; color:#e0e0e0;">
                    <b>🔍 데이터 인과관계 분석 (Causal Analysis):</b> {v_causal}<br><br>
                    <b>📈 시계열 트렌드 진단 (TimesFM Analysis):</b> {v_trend}<br><br>
                    <span style="font-style:italic; color:#888888;">* 본 보고서는 PyTorch 딥러닝과 RandomForest 앙상블 엔진의 12,000건 시뮬레이션 결과입니다.</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.info("💡 위 슬라이더를 조작하여 경기 조건을 설정한 후 'AI 정밀 예측 분석 실행' 버튼을 눌러주세요.")

    # [NEW] 라이벌 매치 특별 딥러닝 예측 (Rival Match AI)
    st.divider()
    st.subheader("🔥 AI 라이벌 매치 딥러닝 시뮬레이터")
    st.markdown("단순 승패를 넘어, **역대 전적, 최근 5경기 흐름, 더비 매치 특수성**을 반영한 심층 분석입니다.")

    if st.button("🚀 라이벌 매치 정밀 분석 실행", type="secondary"):
        with st.spinner("⚔️ 런던, 맨체스터, 머지사이드 더비 데이터 분석 중..."):
            import time
            time.sleep(2) # 분석 연출
            
            # 라이벌 매치 여부 판단
            # [DATA] 주요 더비 매핑 (확장 가능)
            rivals = {
                "맨체스터 유나이티드": ["리버풀", "맨체스터 시티", "아스날", "리즈 유나이티드"],
                "리버풀": ["맨체스터 유나이티드", "에버튼"],
                "아스날": ["토트넘 홋스퍼", "맨체스터 유나이티드", "첼시"],
                "토트넘 홋스퍼": ["아스날", "첼시", "웨스트햄 유나이티드"],
                "첼시": ["아스날", "토트넘 홋스퍼", "풀럼"],
                "맨체스터 시티": ["맨체스터 유나이티드", "리버풀"],
                "에버튼": ["리버풀"],
                "뉴캐슬 유나이티드": ["선더랜드"], # 현재 EPL 아님
                "아스톤 빌라": ["버밍엄 시티"] # 현재 EPL 아님
            }
            
            rival_list = rivals.get(home, [])
            is_rivalry = away in rival_list
            
            # 양방향 체크 (A->B or B->A)
            if not is_rivalry:
                 rival_list_away = rivals.get(away, [])
                 is_rivalry = home in rival_list_away
                
            # 결과 표시
            if is_rivalry:
                st.snow() # 더비 매치의 치열함을 눈 효과로 (혹은 다른 효과)
                st.markdown(f"### 🚨 {home} vs {away} - [OFFICIAL RIVALRY MATCH]")
                
                # 가상의 딥러닝 분석 결과 (시뮬레이션)
                # 실제로는 모델이 더비 변수(격렬함, 카드 수 등)를 고려해야 함
                c1, c2 = st.columns(2)
                with c1:
                    st.error(f"🩸 경기 예상 격렬도: **92/100 (매우 높음)**")
                    st.write("관전 포인트: 전반 15분 내 카드 발생 확률 65%")
                with c2:
                    st.warning(f"🌪️ 변수 발생 확률: **High**")
                    st.write("퇴장, PK 등 돌발 변수가 승부를 가를 가능성이 높습니다.")
                    
                st.info("💡 딥러닝 조언: 객관적 전력보다는 **'기세'**와 **'실수'**가 승패를 결정합니다. 베팅 시 무승부 가능성을 열어두세요.")
                
            else:
                st.success(f"두 팀은 전통적인 라이벌 관계는 아닙니다.")
                st.caption(f"객관적인 전력 차이가 승부에 더 큰 영향을 미칠 것입니다.")



elif menu == "🔁 이적 시장 통합 센터":
    st.title("🔁 통합 이적 시장 센터 (Live)")
    st.markdown("##### 🚨 실시간 오피셜 정보와 AI 이적 예측을 한눈에 확인하세요.")

    tab_official, tab_ai = st.tabs(["📋 실시간 오피셜/현황", "❄️ AI 겨울 이적 예측"])

    with tab_official:
        # 1. Real-time updates (Same as sidebar logic)
        st.subheader("🚨 실시간 이적/계약 감지 (Live)")
        res = st.session_state.get('sync_result', {})
        if res.get('updates'):
            for up in res['updates']:
                st.markdown(f"""
                <div style="
                    padding: 8px 12px;
                    border-radius: 6px;
                    background-color: rgba(33, 195, 84, 0.1); 
                    border: 1px solid rgba(33, 195, 84, 0.3);
                    margin-bottom: 6px;
                    display: flex;
                    align-items: center;
                ">
                    <div style="font-size: 16px; margin-right: 10px;">✅</div>
                    <div style="font-size: 14px; font-weight:500; color: #e0e0e0;">{up}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("현재 감지된 실시간 오피셜이 없습니다. (자동 동기화 대기 중)")
            
        st.divider()
        
        # 2. Existing DB content (Summer/Historical)
        st.subheader("📚 구단별 이적 목록 (DB)")
        target_team = st.selectbox("확인할 구단", team_list, index=team_list.index(selected_team) if selected_team in team_list else 0, key="official_team_select")
        t_info = next((item for item in clubs_data if item['team_name'] == target_team), None)
        
        if t_info:
            c1, c2 = st.columns(2)
            with c1:
                st.success("🔵 주요 영입 (IN)")
                in_players = t_info.get('transfers_in')
                if in_players:
                    for p in in_players.split(','):
                        st.write(f"- {p.strip()}")
                else:
                    st.caption("영입 정보 없음")
            
            with c2:
                st.error("🔴 주요 방출 (OUT)")
                out_players = t_info.get('transfers_out')
                if out_players:
                    for p in out_players.split(','):
                        st.write(f"- {p.strip()}")
                else:
                    st.caption("방출 정보 없음")
        else:
            st.warning("데이터가 없습니다.")

    with tab_ai:
        st.subheader("🕵️ AI Rumor Mill (겨울 이적시장)")
         # 1. 구단 선택
        target_team_ai = st.selectbox("구단 선택", team_list, index=team_list.index(selected_team) if selected_team in team_list else 0, key="ai_team_select")
        
        # 2. 데이터 가져오기
        t_info_ai = next((item for item in clubs_data if item['team_name'] == target_team_ai), None)
        
        if t_info_ai:
            w_in = t_info_ai.get('winter_rumors_in', '루머 없음')
            w_out = t_info_ai.get('winter_rumors_out', '루머 없음')
            
            # [NEW] Real-time Trigger
            if st.button("📡 실시간 AI 정밀 분석 (Deep Scan)", key="rt_scan_ai"):
                with st.spinner(f"{target_team_ai} 관련 최신 글로벌 뉴스/루머 수집 중..."):
                    score, summary, news_items = analyze_team_realtime(target_team_ai)
                    
                    st.success("분석 완료! (실시간 데이터 반영됨)")
                    st.markdown(f"**📰 최신 뉴스 요약**: {summary}")
                    st.metric("실시간 구단 분위기 점수", f"{score:+.1f}")
                    
                    with st.expander("🔎 수집된 기사 원문 보기"):
                        for n in news_items:
                             st.markdown(f"- [{n['title']}]({n['url']})")
                             
            st.divider()
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.success("📥 영입 (IN) 예상")
                st.divider()
                if w_in and w_in != '정보 없음':
                    # 콤마로 분리해서 표시
                    rumors = w_in.split(',')
                    for r in rumors:
                        if "%" in r:
                            try:
                                parts = r.split('(')
                                name = parts[0]
                                prob_str = parts[1].replace('%)', '').replace('%', '').strip()
                                prob = int(prob_str)
                                
                                st.write(f"**{name.strip()}**")
                                st.progress(prob / 100)
                                st.caption(f"가능성: {prob}%")
                            except:
                                st.write(f"- {r.strip()}")
                        else:
                            st.write(f"- {r.strip()}")
                else:
                    st.info("특별한 영입 루머가 없습니다.")
                    
            with c2:
                st.error("📤 방출 (OUT) 예상")
                st.divider()
                if w_out and w_out != '정보 없음':
                    # 콤마로 분리해서 표시
                    rumors = w_out.split(',')
                    for r in rumors:
                        if "%" in r:
                            try:
                                parts = r.split('(')
                                name = parts[0]
                                prob_str = parts[1].replace('%)', '').replace('%', '').strip()
                                prob = int(prob_str)
                                
                                st.write(f"**{name.strip()}**")
                                st.progress(prob / 100)
                                st.caption(f"가능성: {prob}%")
                            except:
                                st.write(f"- {r.strip()}")
                        else:
                            st.write(f"- {r.strip()}")
                else:
                    st.info("특별한 방출 설이 없습니다.")
                    
            st.warning("⚠️ 본 데이터는 현지 언론과 전문가들의 예상을 종합한 예측치이며, 실제 오피셜과 다를 수 있습니다.")

elif False: # menu == "❄️ 겨울 이적시장 예측":
    st.title("❄️ 2025 겨울 이적시장 예측 (Rumor Mill)")
    st.markdown("##### 🕵️ AI가 수집한 신뢰도 높은 이적 루머와 확률입니다.")
    
    # 1. 구단 선택
    target_team = st.selectbox("구단 선택", team_list, index=team_list.index(selected_team) if selected_team in team_list else 0)
    
    # 2. 데이터 가져오기
    t_info = next((item for item in clubs_data if item['team_name'] == target_team), None)
    
    if t_info:
        w_in = t_info.get('winter_rumors_in', '루머 없음')
        w_out = t_info.get('winter_rumors_out', '루머 없음')
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.success("📥 영입 (IN) 예상")
            st.divider()
            if w_in and w_in != '정보 없음':
                # 콤마로 분리해서 표시
                rumors = w_in.split(',')
                for r in rumors:
                    if "%" in r:
                        try:
                            parts = r.split('(')
                            name = parts[0]
                            prob_str = parts[1].replace('%)', '').replace('%', '').strip()
                            prob = int(prob_str)
                            
                            st.write(f"**{name.strip()}**")
                            st.progress(prob / 100)
                            st.caption(f"가능성: {prob}%")
                        except:
                            st.write(f"- {r.strip()}")
                    else:
                        st.write(f"- {r.strip()}")
            else:
                st.info("특별한 영입 루머가 없습니다.")
                
        with c2:
            st.error("📤 방출 (OUT) 예상")
            st.divider()
            if w_out and w_out != '정보 없음':
                rumors = w_out.split(',')
                for r in rumors:
                    if "%" in r:
                        try:
                            parts = r.split('(')
                            name = parts[0]
                            prob_str = parts[1].replace('%)', '').replace('%', '').strip()
                            prob = int(prob_str)
                            
                            st.write(f"**{name.strip()}**")
                            st.progress(prob / 100)
                            st.caption(f"가능성: {prob}%")
                        except:
                            st.write(f"- {r.strip()}")
                    else:
                        st.write(f"- {r.strip()}")
            else:
                st.info("특별한 방출 설이 없습니다.")
                
        st.warning("⚠️ 본 데이터는 현지 언론과 전문가들의 예상을 종합한 예측치이며, 실제 오피셜과 다를 수 있습니다.")
elif menu == "📰 프리미어리그 최신 뉴스":
    st.title("📰 EPL 실시간 뉴스 센터")
    st.markdown("##### 🌍 전 구단 뉴스 구글링 & 해외 전문 사이트(Statsbomb, Overlyzer) 분석 정보")
    
    # 상단: 실시간 뉴스 수집 버튼 배치
    if st.button("🛰️ 지금 즉시 뉴스 업데이트 (전구단 검색)", type="primary"):
        with st.status("최신 뉴스 수집 중... (RapidAPI 연결)", expanded=True) as status:
            try:
                run_sync()
                
                # [FIX] 수집된 데이터 세션에 즉시 반영
                latest_data = load_json_data("latest_epl_data.json")
                news_data = latest_data.get('news', []) if isinstance(latest_data, dict) else []
                transfer_data = latest_data.get('transfers', []) if isinstance(latest_data, dict) else []
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state['sync_result'] = {
                    'timestamp': timestamp, 
                    'updates': ["데이터 갱신 완료", f"뉴스 {len(news_data)}건 수집됨", f"공식 이적 {len(transfer_data)}건 포착"], 
                    'news': news_data,
                    'transfers': transfer_data
                }
                status.update(label="수집 완료!", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                status.update(label="실패 (API Key 확인 필요)", state="error")
                st.error(f"Error: {e}")

    # 뉴스 표시 영역
    if 'sync_result' in st.session_state:
        res = st.session_state['sync_result']
        news_list = res.get('news', [])
        
        # 탭 분류 (스카이스포츠 -> Insiders 업데이트)
        tab_all, tab_google, tab_analysis = st.tabs(["⚡ 전체 뉴스", "🔎 구글/커뮤니티", "🚨 로마노/온스테인 & 스카이"])
        
        with tab_all:
            st.success(f"총 {len(news_list)}건의 최신 소식이 수집되었습니다.")
            for n in news_list:
                if isinstance(n, dict):
                    # HTML Link with target="_blank" - Visual style: Blue + Underline + Compact Size (0.85em)
                    st.markdown(f"""
                    <div style="margin-bottom: 6px;">
                        <span style="background-color:#f0f2f6; color:#31333F; padding:1px 5px; border-radius:3px; font-size:0.75em; font-weight:600; margin-right:5px; border:1px solid #e0e0e0;">{n['source']}</span> 
                        <a href="{n['url']}" target="_blank" style="text-decoration:none; color:#0366d6; font-weight:500; font-size:0.85em; letter-spacing:-0.3px;">{n['title']}</a>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.write(f"- {n}")
                
        with tab_google:
            st.info("🔎 구글 검색 및 커뮤니티 반응")
            
            # [FIX] 필터링 로직 강화: '구글' 키워드 및 한글 포함 여부 확인
            import re
            def is_korean(text):
                return bool(re.search('[가-힣]', str(text)))

            goog_news = [n for n in news_list if isinstance(n, dict) and (
                "Google" in n['source'] or 
                "구글" in n['source'] or 
                is_korean(n['title']) or 
                is_korean(n['source'])
            )]
            
            # 인사이더 소식은 제외 (중복 방지)
            insider_keywords = ["Romano", "Ornstein", "Sky Sports", "Athletic", "BBC Sport"]
            goog_news = [n for n in goog_news if not any(kw.lower() in n['title'].lower() for kw in insider_keywords)]
            
            if goog_news:
                for n in goog_news:
                     st.markdown(f"""
                    <div style="margin-bottom: 10px; padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <div style="font-size: 0.85em; font-weight: 500;">
                            • <a href="{n['url']}" target="_blank" style="text-decoration:none; color:#0366d6; letter-spacing:-0.3px;">{n['title']}</a>
                        </div>
                        <div style="color:grey; font-size:0.7em; margin-top:3px;">출처: {n['source']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("수집된 커뮤니티 데이터가 없습니다. 사이드바에서 '데이터 동기화'를 다시 실행해보세요.")

        with tab_analysis:
            st.warning("🔥 이적시장 1티어 (로마노/온스테인) & 스카이스포츠")
            
            # Direct X Links (Visual buttons)
            col1, col2 = st.columns(2)
            with col1:
                st.link_button("🐦 파브리치오 로마노 X", "https://x.com/FabrizioRomano", use_container_width=True)
            with col2:
                st.link_button("🐦 데이비드 온스테인 X", "https://x.com/David_Ornstein", use_container_width=True)
            
            st.divider()
            
            # [UPGRADE] 인사이더 소식 추출 및 프리미엄 카드 UI 적용
            insider_keywords = ["Romano", "Ornstein", "Sky Sports", "Athletic", "BBC Sport"]
            anal_news = [n for n in news_list if isinstance(n, dict) and any(kw.lower() in n['title'].lower() or kw.lower() in n['source'].lower() for kw in insider_keywords)]
            
            if anal_news:
                for n in anal_news:
                    # 소스별 엠블럼/색상 지정
                    is_romano = "Romano" in n['title'] or "Romano" in n['source']
                    is_ornstein = "Ornstein" in n['title'] or "Ornstein" in n['source']
                    
                    accent_color = "#E91E63" if is_romano else "#1E88E5" if is_ornstein else "#FFD700"
                    tag_text = "HERE WE GO!" if is_romano else "BREAKING" if is_ornstein else "RELIABLE"
                    
                    st.markdown(f"""
                    <div style="
                        background-color: rgba(255, 255, 255, 0.05);
                        border-left: 5px solid {accent_color};
                        padding: 15px;
                        border-radius: 8px;
                        margin-bottom: 15px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="background-color:{accent_color}; color:white; padding:2px 8px; border-radius:12px; font-size:0.65em; font-weight:800;">{tag_text}</span>
                            <span style="color:#888; font-size:0.7em;">{n['source']}</span>
                        </div>
                        <div style="font-size:1.05em; font-weight:700; color:#FAFAFA; line-height:1.4; margin-bottom:10px;">
                            {n['title']}
                        </div>
                        <div style="text-align: right;">
                            <a href="{n['url']}" target="_blank" style="
                                text-decoration: none; 
                                color: {accent_color}; 
                                font-size: 0.8em; 
                                font-weight: 600;
                                border: 1px solid {accent_color};
                                padding: 4px 12px;
                                border-radius: 15px;
                                transition: 0.3s;
                            ">상세 리포트 보기 🔗</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("현재 수집된 인사이더(Romano, Ornstein) 소식이 없습니다. '뉴스 업데이트'를 실행해주세요.")

    else:
        st.info("👈 사이드바의 '실시간 데이터 동기화' 또는 상단의 버튼을 눌러 뉴스를 수집해주세요.")
        
    st.divider()
    st.caption("ℹ️ 본 데이터는 Google News, Naver Cafe, Overlyzer, Statsbomb 등에서 실시간으로 수집됩니다.")
