import http.client
import json
import os
import pandas as pd
from datetime import datetime

# ==========================================
# 🔧 설정 (Configuration)
# ==========================================
API_HOST = "v3.football.api-sports.io"
API_KEY = os.getenv("RAPIDAPI_KEY", "") # GitHub Secrets 또는 환경변수에서 로드

DATA_DIR = os.path.join(os.path.dirname(__file__), "data/advanced")
SEASON = 2024
LEAGUE_ID = 39

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def fetch_from_api(endpoint: str):
    if not API_KEY:
        print("❌ Error: API Key is missing.")
        return None
    conn = http.client.HTTPSConnection(API_HOST)
    headers = { 'x-apisports-key': API_KEY }
    try:
        conn.request("GET", endpoint, headers=headers)
        res = conn.getresponse()
        if res.status != 200: return None
        return json.loads(res.read().decode("utf-8"))
    except: return None
    finally: conn.close()

def collect_advanced_team_stats():
    """
    AI 학습을 위한 구단별 정밀 스탯 수집 (Causal AI & TimesFM 기반)
    """
    print("🚀 [Deep Scan] 구단별 정밀 지표 수집 중...")
    
    # 1. 우선 순위표를 가져와서 팀 ID 목록 확보
    standings = fetch_from_api(f"/v3/standings?season={SEASON}&league={LEAGUE_ID}")
    if not standings or not standings.get('response'): 
        print("❌ 순위표 수집 실패")
        return
        
    teams = standings['response'][0]['league']['standings'][0]
    
    advanced_db = []
    
    # 무료 플랜 API 호출 제한(하루 100회)을 고려하여 상위 5팀만 우선 정밀 샘플링 (테스트용)
    # 실제 운영 시에는 매일 5팀씩 순차적으로 업데이트하거나 유료 플랜 활용
    for team_entry in teams[:5]:
        team_id = team_entry['team']['id']
        team_name = team_entry['team']['name']
        print(f"📡 {team_name} (ID: {team_id}) 정밀 데이터 추출...")
        
        # 구단 통계 (공격/수비 지표)
        stats = fetch_from_api(f"/v3/teams/statistics?season={SEASON}&league={LEAGUE_ID}&team={team_id}")
        if stats and stats.get('response'):
            s = stats['response']
            refined = {
                "team_name": team_name,
                "team_id": team_id,
                "goals_scored": s['goals']['for']['total']['total'],
                "goals_conceded": s['goals']['against']['total']['total'],
                "clean_sheets": s['clean_sheet']['total'],
                "failed_to_score": s['failed_to_score']['total'],
                "form": s['form'], # 최근 흐름 (TimesFM 입력용)
                "avg_possession": 50, # 예시 (실제 Fixture Stats에서 평균내야 함)
                "last_updated": datetime.now().strftime("%Y-%m-%d")
            }
            advanced_db.append(refined)
            
    # JSON 저장
    output_path = os.path.join(DATA_DIR, "team_advanced_stats.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(advanced_db, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 정밀 데이터 저장 완료: {output_path}")

if __name__ == "__main__":
    collect_advanced_team_stats()
