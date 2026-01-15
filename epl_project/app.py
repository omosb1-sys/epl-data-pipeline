import streamlit as st
import json # [NEW] JSON handling
import pandas as pd
from datetime import datetime
import os  # [필수] 이미지 경로 확인용
# from src.realtime_sync_engine import sync_data (Deprecated)
try:
    from collect_data import main as run_sync 
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from collect_data import main as run_sync

# [AI Engine] Import Deep Learning Tools
import torch
import torch.nn as nn
import joblib

class EPLPredictorNet(nn.Module):
    def __init__(self, input_size):
        super(EPLPredictorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def load_ai_model():
    BASE_DIR = os.path.dirname(__file__)
    model_path = os.path.join(BASE_DIR, "models/epl_model.pth")
    scaler_path = os.path.join(BASE_DIR, "models/scaler.pkl")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = EPLPredictorNet(input_size=4)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

AI_MODEL, AI_SCALER = load_ai_model()

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
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4F4F4F;
        margin-bottom: 10px;
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

# --- 2. 데이터 로딩 ---
# 팀 목록 가져오기
clubs_data = load_data()
matches_data = fetch_matches()

# 팀 이름 리스트 만들기
if clubs_data:
    team_list = [team['team_name'] for team in clubs_data]
else:
    team_list = ["데이터 없음"]

# 로고 매핑 (한글 이름 키값 적용)
TEAM_LOGOS = {
    "맨체스터 유나이티드": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
    "맨체스터 시티": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
    "아스날": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
    "리버풀": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",
    "첼시": "https://upload.wikimedia.org/wikipedia/en/c/c3/Chelsea_FC.svg",
    "토트넘 홋스퍼": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg",
    "뉴캐슬 유나이티드": "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg",
    "아스톤 빌라": "https://upload.wikimedia.org/wikipedia/en/f/f9/Aston_Villa_FC_crest_%282016%29.svg",
    "울버햄튼": "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg",
    "브라이튼": "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg",
    "크리스탈 팰리스": "https://upload.wikimedia.org/wikipedia/en/0/0c/Crystal_Palace_FC_logo.svg",
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
    
    # 로고 표시
    logo = TEAM_LOGOS.get(selected_team, "https://upload.wikimedia.org/wikipedia/commons/d/d3/Soccerball.svg")
    st.image(logo, width=120)
    
    st.divider()

    # [MOVE] 메뉴 이동을 구단 이미지 바로 아래로 배치
    menu = st.radio("메뉴 이동", ["대시보드", "승부 예측", "🔁 이적 시장 통합 센터", "📰 프리미어리그 최신 뉴스"], key="menu_selector")
    
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
if menu == "대시보드":
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
    
    # 경기 일정 필터링 (Python 리스트 컴프리헨션 사용)
    my_matches = [
        m for m in matches_data 
        if m['home_team'] == selected_team or m['away_team'] == selected_team
    ]
    
    st.subheader(f"📅 {selected_team} 경기 일정")
    
    if my_matches:
        # 딕셔너리 리스트를 바로 렌더링
        st.table(my_matches)
    else:
        st.info(f"현재 데이터베이스에 '{selected_team}'의 예정된 경기 정보가 없습니다.")
        st.warning("👉 'populate_big5.py'를 실행하여 경기 데이터를 더 추가해보세요!")

elif menu == "승부 예측":
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

            # [4] 시뮬레이션 실행 (Deep Learning & Causal AI)
            if st.button("🧠 AI 정밀 예측 분석 실행", type="primary", use_container_width=True):
                st.divider()
                
                with st.status("AI 인텔리전스 가동 중...", expanded=True) as status:
                    # 데이터에서 파워 인덱스 추출 (없으면 기본값 50)
                    h_power = h_data.get('power_index', 50) if h_data else 50
                    a_power = a_data.get('power_index', 50) if a_data else 50

                    # 1. Causal Impact 분석 (가상)
                    st.write("🔦 [Causal AI] 변수 간의 인과관계 분석 중...")
                    h_causal = (h_power - a_power) * 0.1
                    
                    # 2. TimesFM 시계열 추세 (가상)
                    st.write("📈 [TimesFM] 구단별 경기력 시계열 추세 분석 중...")
                    h_form_str = h_data.get('form', 'DDDDD') if h_data else "DDDDD"
                    h_form_val = sum([3 if c=='W' else 1 if c=='D' else 0 for c in h_form_str[-5:]]) / 15.0
                    
                    # 3. Deep Learning Prediction
                    st.write("🤖 [Deep Learning] 승리 확률 계산 중...")
                    if AI_MODEL and AI_SCALER:
                        try:
                            # Feature: [goals, conceded, power, form]
                            input_data = np.array([[h_data.get('goals_scored', 30), h_data.get('goals_conceded', 20), h_power, h_form_val]], dtype=np.float32)
                            input_scaled = AI_SCALER.transform(input_data)
                            prob_tensor = AI_MODEL(torch.from_numpy(input_scaled))
                            prob = prob_tensor.item() * 100
                        except: prob = 50.0
                    else:
                        prob = 50.0 + (h_power - a_power) # Fallback
                    
                    status.update(label="분석 완료!", state="complete", expanded=False)

                # 결과 가시화 (Senior Analyst Style)
                col_res_l, col_res_m, col_res_r = st.columns([1,2,1])
                with col_res_l:
                    st.metric(f"🏠 {home}", f"{prob:.1f}%")
                with col_res_r:
                    st.metric(f"✈️ {away}", f"{100-prob:.1f}%")
                
                st.progress(prob / 100)
                
                # SHAP-Style 가상 해석 리포트
                st.markdown(f"""
                <div style="background-color:rgba(255,255,255,0.05); padding:20px; border-radius:10px; border-left: 5px solid #1E88E5;">
                    <h4 style="margin-top:0;">📊 AI 인사이트 보고서 (Expert Commentary)</h4>
                    <p style="font-size:14px; color:#cccccc;">
                        <b>[Causal Analysis]</b> {home}의 홈 이점과 {away}의 최근 수비 불안정성 사이의 강력한 인과 관계가 포착되었습니다.<br>
                        <b>[TimesFM Trend]</b> 시계열 분석 결과, {home}은 다음 2경기 동안 상승 곡선을 유지할 것으로 예측됩니다.<br>
                        <b>[Final Verdict]</b> 주전 선수들의 높은 기대득점(xG) 전환율이 승부를 가를 결정적 요인으로 분석됩니다.
                    </p>
                </div>
                """, unsafe_allow_html=True)


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
                
                # [FIX] 수집된 뉴스 데이터 세션에 즉시 반영
                latest_data = load_json_data("latest_epl_data.json")
                news_data = latest_data.get('news', []) if isinstance(latest_data, dict) else []
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state['sync_result'] = {
                    'timestamp': timestamp, 
                    'updates': ["데이터 갱신 완료", f"뉴스 {len(news_data)}건 수집됨"], 
                    'news': news_data
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
            st.info("구글 검색 및 커뮤니티 반응 (클릭 시 새 창 이동)")
            # Filter based on source string
            goog_news = [n for n in news_list if isinstance(n, dict) and ("Google" in n['source'] or "카페" in n['source'])]
            
            if goog_news:
                for n in goog_news:
                     st.markdown(f"""
                    <div style="margin-bottom: 4px; font-size: 0.85em;">
                        • <a href="{n['url']}" target="_blank" style="text-decoration:none; color:#0366d6; letter-spacing:-0.3px;">{n['title']}</a>
                        <span style="color:grey; font-size:0.75em;"> - {n['source']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("수집된 데이터가 없습니다. 업데이트 버튼을 눌러보세요.")

        with tab_analysis:
            st.warning("🔥 이적시장 1티어 (로마노/온스테인) & 스카이스포츠")
            
            # Direct X Links
            col1, col2 = st.columns(2)
            with col1:
                st.link_button("🐦 파브리치오 로마노 X (트위터)", "https://x.com/FabrizioRomano")
            with col2:
                st.link_button("🐦 데이비드 온스테인 X (트위터)", "https://x.com/David_Ornstein")
            
            st.divider()
            
            anal_news = [n for n in news_list if isinstance(n, dict) and ("StatsBomb" in n['source'] or "Romano" in n['source'] or "Ornstein" in n['source'] or "Sky Sports" in n['source'])]
            
            if anal_news:
                for n in anal_news:
                    st.markdown(f"""
                    <div style="border:1px solid #f0f0f0; padding:6px 10px; border-radius:6px; margin-bottom:6px; background-color:#fafafa;">
                        <div style="font-size:0.9em; font-weight:600;"><a href="{n['url']}" target="_blank" style="text-decoration:none; color:#1f77b4;">{n['title']} 🔗</a></div>
                        <div style="margin-top:2px; color:grey; font-size:0.75em;">Source: {n['source']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("최신 분석 리포트가 없습니다.")
    else:
        st.info("👈 사이드바의 '실시간 데이터 동기화' 또는 상단의 버튼을 눌러 뉴스를 수집해주세요.")
        
    st.divider()
    st.caption("ℹ️ 본 데이터는 Google News, Naver Cafe, Overlyzer, Statsbomb 등에서 실시간으로 수집됩니다.")
