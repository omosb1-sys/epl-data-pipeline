import http.client
import json
import os
import time
from datetime import datetime
import requests # [NEW] News scraping
from bs4 import BeautifulSoup # [NEW] News scraping

# ==========================================
# 🔧 설정 (Configuration)
# ==========================================
# ==========================================
# 🔧 설정 (Configuration)
# ==========================================
# [UPDATE] RapidAPI Deprecation -> Official Direct API
API_HOST = "v3.football.api-sports.io"
# API 키는 환경 변수에서 가져오거나, 없으면 빈 문자열 (GitHub Secrets 사용 권장)
API_KEY = os.getenv("RAPIDAPI_KEY", "") 

# 저장 경로
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "latest_epl_data.json")

# EPL League ID (39 = Premier League)
LEAGUE_ID = 39
SEASON = 2024 # 2024-2025 Season

def fetch_from_api(endpoint):
    """
    API-Football Direct (v3)를 통해 데이터를 가져오는 함수
    """
    if not API_KEY:
        print("❌ Error: API Key is missing.")
        return None

    conn = http.client.HTTPSConnection(API_HOST)
    headers = {
        'x-apisports-key': API_KEY
    }
    
    try:
        print(f"📡 Requesting: {endpoint}...")
        conn.request("GET", endpoint, headers=headers)
        res = conn.getresponse()
        data = res.read()
        
        if res.status != 200:
            print(f"❌ API Error {res.status}: {res.reason}")
            return None
            
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return None
    finally:
        conn.close()

def scrape_epl_news():
    """Google News RSS를 통해 EPL 최신 뉴스를 가져옵니다 (매우 안정적)"""
    print("📡 Fetching news from Google News RSS...")
    # 영문 뉴스 (Premier League 검색)
    url = "https://news.google.com/rss/search?q=Premier+League+News&hl=en-GB&gl=GB&ceid=GB:en"
    news_list = []
    
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, 'xml') # XML 파싱
        
        items = soup.find_all('item', limit=15)
        for item in items:
            title = item.title.text
            link = item.link.text
            source = item.source.text if item.source else "Google News"
            
            news_list.append({
                "source": source,
                "title": title,
                "url": link
            })
            
        # 한글 뉴스 추가 (프리미어리그 검색)
        url_ko = "https://news.google.com/rss/search?q=프리미어리그&hl=ko&gl=KR&ceid=KR:ko"
        res_ko = requests.get(url_ko, timeout=10)
        soup_ko = BeautifulSoup(res_ko.text, 'xml')
        items_ko = soup_ko.find_all('item', limit=10)
        for item in items_ko:
            news_list.append({
                "source": item.source.text if item.source else "구글 뉴스",
                "title": item.title.text,
                "url": item.link.text
            })

        print(f"✅ Total News collected: {len(news_list)} items")
    except Exception as e:
        print(f"⚠️ News fetching failed: {e}")
        
    return news_list

def main():
    print("🚀 [EPL Data Robot] Starting data collection...")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    final_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "season": SEASON,
        "standings": [],
        "fixtures": [],
        "top_scorers": [],
        "news": [] # [NEW]
    }
    
    # ... API 호출부 (생략) ...
    # 1. Standings
    # 2. Fixtures
    # (위의 코드가 계속 있다고 가정)
    
    # [FIX] main 함수의 흐름을 유지하기 위해 기존 코드를 정확히 매칭해서 넣어줍니다.
    # 1. Standings
    standings_data = fetch_from_api(f"/v3/standings?season={SEASON}&league={LEAGUE_ID}")
    if standings_data and standings_data.get('response'):
        final_data['standings'] = standings_data['response'][0]['league']['standings'][0]
        print(f"✅ Standings collected.")

    # 2. Fixtures
    current_round_resp = fetch_from_api(f"/v3/fixtures/rounds?season={SEASON}&league={LEAGUE_ID}&current=true")
    if current_round_resp and current_round_resp.get('response'):
        current_round = current_round_resp['response'][0]
        fixtures_data = fetch_from_api(f"/v3/fixtures?season={SEASON}&league={LEAGUE_ID}&round={current_round}")
        if fixtures_data and fixtures_data.get('response'):
            final_data['fixtures'] = fixtures_data['response']
            print(f"✅ Fixtures collected.")

    # 4. News Scraping
    final_data['news'] = scrape_epl_news()

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)
        
    print(f"💾 Data saved to: {OUTPUT_FILE}")
    print("✨ Mission Complete!")

if __name__ == "__main__":
    main()
