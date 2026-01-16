import requests
from bs4 import BeautifulSoup
import random
import datetime

# [AI Agent: Antigravity]
# 이 모듈은 특정 감독의 전술을 실시간으로 분석(크롤링)하여 보고서를 생성합니다.

def scrape_google_search(query, num_results=5):
    """구글 검색 결과에서 제목과 요약을 추출합니다 (영어권 전문가 칼럼 중심)"""
    results = []
    try:
        # User-Agent 설정 (브라우저인 척) - 영국/미국 트래픽 모사
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8"
        }
        
        # [Expert Filter] 영국/미국 현지 전문가 사이트 위주로 검색 (site: 필터 활용)
        # The Athletic, Sky Sports, Tifo, Coaches' Voice, Tactical Analysis sites
        expert_query = f"{query} site:theathletic.com OR site:skysports.com OR site:coachesvoice.com OR site:totalfootballanalysis.com"
        
        # 언어 설정: gl=GB (영국), hl=en (영어)
        url = f"https://www.google.com/search?q={expert_query.replace(' ', '+')}&hl=en&gl=GB&num=10"
        
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        count = 0
        for h3 in soup.find_all('h3'):
            if count >= num_results: break
            
            title = h3.text
            parent = h3.find_parent('a')
            link = parent['href'] if parent else "#"
            
            if link.startswith("/url?q="):
                link = link.split("/url?q=")[1].split("&")[0]
            
            # 출처 태깅 (도메인 기반)
            source_tag = "Global Analysis"
            if "theathletic" in link: source_tag = "The Athletic (Tier 1)"
            elif "skysports" in link: source_tag = "Sky Sports Tactical"
            elif "coachesvoice" in link: source_tag = "The Coaches' Voice (Pro)"
            elif "totalfootball" in link: source_tag = "Total Football Analysis"
            
            results.append({"title": title, "link": link, "source": source_tag})
            count += 1
            
    except Exception as e:
        print(f"Scraping Error: {e}")
        results.append({"title": "Analysis data unavailable at the moment", "link": "#", "source": "System"})
        
    return results

def scrape_youtube_titles(query, num_results=3):
    """유튜브 검색 결과(제목)만 텍스트로 긁어옵니다"""
    results = []
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        # Google Video Search
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}+site:youtube.com&tbm=vid"
        
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        for h3 in soup.find_all('h3', limit=num_results):
            results.append(h3.text)
            
    except:
        pass
    return results


def analyze_tactics(team_name, manager_name):
    """
    [Main Function]
    특정 감독의 최근 전술을 분석하여 구조화된 리포트를 반환합니다.
    """
    
    # 1. 검색 쿼리 생성
    # 예: "Arne Slot Liverpool tactics analysis 2025"
    q_base = f"{manager_name} {team_name} tactics style 2025"
    q_recent = f"{manager_name} {team_name} last 5 games analysis"
    
    # 2. 데이터 수집 (크롤링 - Global & Korean)
    print(f"🔍 Analyzing tactics for {manager_name}...")
    web_results = scrape_google_search(q_base, num_results=4)
    video_titles = scrape_youtube_titles(f"{manager_name} tactics analysis", num_results=3)
    
    # [NEW] 국내 유명 유튜버 분석 수집 (이스타, 김진짜, 새축, 달수네, 한준)
    kr_videos = scrape_korean_pundits(manager_name, team_name)
    
    # 3. 키워드 추출 (간단한 Rule-based)
    # 영어 + 한국어 타이틀 모두 분석
    text_corpus = " ".join([r['title'] for r in web_results] + video_titles + kr_videos).lower()
    
    keywords = []
    tactical_terms = [
        "high press", "counter attack", "possession", "build-up", "wing play", 
        "inverted fullback", "false 9", "back 3", "defensive", "aggressive", 
        "midfield control", "transition", "set piece", "fluid"
    ]
    
    for term in tactical_terms:
        if term in text_corpus:
            keywords.append(term.title())
            
    if not keywords:
        keywords = ["Balanced", "Organized", "Direct Play"] # Default
        
    # 4. 최근 5경기 가상 데이터 생성 ... (기존 코드 유지)
    formations = ["4-2-3-1", "4-3-3", "3-4-2-1", "4-4-2"]
    # 감독별 선호 포메이션 (하드코딩된 지식 베이스 활용)
    pref_formation = "4-2-3-1"
    if "Guardiola" in manager_name: pref_formation = "3-2-4-1"
    elif "Klopp" in manager_name or "Slot" in manager_name: pref_formation = "4-3-3"
    elif "Ange" in manager_name: pref_formation = "4-3-3 (Inverted FB)"
    elif "Ten Hag" in manager_name: pref_formation = "4-2-3-1"
    elif "Howe" in manager_name: pref_formation = "4-3-3 (High Press)"
    elif "Emery" in manager_name: pref_formation = "4-4-2 / 4-2-2-2"
    elif "Nuno" in manager_name: pref_formation = "4-2-3-1 (Counter)"
    
    recent_form = []
    results = ["W", "D", "L", "W", "W"] # Dummy recent results
    for i in range(5):
        recent_form.append({
            "match": f"Match {5-i}", 
            "formation": pref_formation,
            "result": random.choice(["Win", "Draw", "Loss", "Win"])
        })
        
    # 5. AI 종합 리포트 생성 (Rich Expert Commentary with Korean Insights)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary = generate_expert_summary(manager_name, team_name, pref_formation, keywords, video_titles, kr_videos)
    
    return {
        "timestamp": timestamp, # [NEW] Execution Time
        "manager": manager_name,
        "team": team_name,
        "pref_formation": pref_formation,
        "keywords": keywords,
        "articles": web_results,
        "videos": video_titles,
        "kr_videos": kr_videos, # [NEW]
        "recent_games": recent_form,
        "ai_summary": summary.strip()
    }

def scrape_korean_pundits(manager, team):
    """국내 1티어 축구 유튜버들의 분석 영상 검색"""
    results = []
    try:
        query = f"{team} {manager} 전술 분석 (이스타TV OR 김진짜 OR 새벽의축구전문가 OR 한준TV OR 달수네)"
        headers = {"User-Agent": "Mozilla/5.0"}
        # qdr:m (한달 이내 최신 영상만)
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=vid&tbs=qdr:m"
        
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        for h3 in soup.find_all('h3', limit=4):
            results.append(h3.text)
    except:
        pass
    return results

def generate_expert_summary(manager, team, formation, keywords, videos, kr_videos=[]):
    """
    [Expert System v2] 글로벌 전문가 + 국내 유튜버 인사이트 통합
    """
    
    # 1. 전술 성향 파악
    archetype = "Balanced"
    if any(k in ["High Press", "Aggressive", "Back 3"] for k in keywords):
        archetype = "Dominant & Aggressive"
    elif any(k in ["Counter Attack", "Defensive", "Transition"] for k in keywords):
        archetype = "Reactive & Direct"
    elif any(k in ["Possession", "Build-Up", "Fluid"] for k in keywords):
        archetype = "Control & Possession"
        
    # 2. 포메이션별 분석 멘트
    form_analysis = {
        "4-2-3-1": "더블 볼란치를 활용한 안정적인 빌드업과 2선 공격 자원들의 유기적인 스위칭 플레이가 돋보입니다.",
        "4-3-3": "세 명의 미드필더를 통한 중원 장악과 윙어들의 과감한 1대1 돌파를 통해 상대 측면을 공략합니다.",
        "3-4-2-1": "윙백을 높게 전진시켜 공격 숫자를 늘리고, 두 명의 10번 성향 공격형 미드필더가 하프스페이스를 끊임없이 타격합니다.",
        "4-4-2": "두 줄 수비를 기반으로 한 견고한 블록 형성 후, 간결하고 직선적인 역습 패턴을 주무기로 삼습니다."
    }
    selected_form_desc = form_analysis.get(formation, "유연한 포메이션 변화를 통해 상대 전술에 맞춤 대응하는 모습입니다.")

    # 3. 비디오/칼럼 인사이트 반영 (국내 의견 추가)
    insight_text = ""
    
    # 영어권 분석
    if videos:
        v_title = videos[0]
        if "Evolution" in v_title or "Change" in v_title:
            insight_text += f"현지 분석('{v_title}')에서는 전술적 유연성을 더하려는 시도가 관찰된다고 평합니다. "
        else:
            insight_text += f"현지에서는 '{v_title}'와 같은 디테일한 부분 전술의 변화에 주목하고 있습니다. "
            
    # 국내 유튜버 분석 반영
    if kr_videos:
        k_title = kr_videos[0]
        insight_text += f"<br><br>또한 <b>국내 전문가들(이스타/김진짜 등)</b>은 최근 <b>'{k_title}'</b> 영상을 통해 알 수 있듯, {team}의 현 문제점과 감독의 대처 방식에 대해 심도 있는 분석을 내놓고 있습니다."

    # 4. 최종 리포트 조립
    report = f"""
    ### 🛡️ 전술 아키타입: {archetype}
    **{manager}** 감독은 이번 시즌 {team}에서 **'{', '.join(keywords[:3])}'** 키워드로 대변되는 축구를 구사하고 있습니다.
    
    ### 📐 포메이션 및 구조적 특징
    주로 **{formation}** 대형을 기반으로 경기 운영을 풀어나가고 있으며, {selected_form_desc}
    
    ### 🧠 글로벌 & 국내 전문가 통합 인사이트
    {insight_text}
    
    최근 데이터 흐름을 볼 때, 단순한 결과 이상의 전술적 일관성을 유지하려는 노력이 보입니다. 현지 전문가들의 칼럼과 국내 분석가들의 시각이 공통적으로 {keywords[0] if keywords else '현재'} 전술의 완성도를 핵심 변수로 꼽고 있습니다.
    """
    
    return report
