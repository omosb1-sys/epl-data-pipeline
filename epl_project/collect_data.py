import http.client
import json
import os
import time
from datetime import datetime

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

def main():
    print("🚀 [EPL Data Robot] Starting data collection...")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    final_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "season": SEASON,
        "standings": [],
        "fixtures": [],
        "top_scorers": []
    }
    
    # 1. Standings (순위표)
    standings_data = fetch_from_api(f"/v3/standings?season={SEASON}&league={LEAGUE_ID}")
    if standings_data and standings_data.get('response'):
        final_data['standings'] = standings_data['response'][0]['league']['standings'][0]
        print(f"✅ Standings collected: {len(final_data['standings'])} teams")
    else:
        print("⚠️ Failed to fetch standings.")

    # 2. Fixtures (경기 일정 - 최근 3경기 & 다음 3경기)
    # Note: 무료 플랜(하루 100회) 절약을 위해 '이번 라운드' 위주로 가져오거나
    # 전체를 가져와서 로컬에서 필터링하는 방식이 좋습니다.
    # 여기서는 '현재 진행 중인 라운드'를 자동으로 찾아서 가져오는 로직을 씁니다.
    
    current_round_resp = fetch_from_api(f"/v3/fixtures/rounds?season={SEASON}&league={LEAGUE_ID}&current=true")
    if current_round_resp and current_round_resp.get('response'):
        current_round = current_round_resp['response'][0]
        print(f"📍 Current Round: {current_round}")
        
        fixtures_data = fetch_from_api(f"/v3/fixtures?season={SEASON}&league={LEAGUE_ID}&round={current_round}")
        if fixtures_data and fixtures_data.get('response'):
            final_data['fixtures'] = fixtures_data['response']
            print(f"✅ Fixtures collected: {len(final_data['fixtures'])} matches")
    
    # 3. Top Scorers (득점 순위) - Optional (비용 절약 위해 가끔 실행 가능)
    # scorers_data = fetch_from_api(f"/v3/players/topscorers?season={SEASON}&league={LEAGUE_ID}")
    # if scorers_data and scorers_data.get('response'):
    #     final_data['top_scorers'] = scorers_data['response']
    #     print(f"✅ Top Scorers collected.")

    # 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)
        
    print(f"💾 Data saved to: {OUTPUT_FILE}")
    print("✨ Mission Complete!")

if __name__ == "__main__":
    main()
