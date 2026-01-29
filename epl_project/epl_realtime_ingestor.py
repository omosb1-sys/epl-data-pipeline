import http.client
import json
import os
import time
from datetime import datetime, timedelta
import pandas as pd
from epl_duckdb_manager import EPLDuckDBManager

# ==========================================
# 🔧 설정 (Configuration)
# ==========================================
API_HOST = "v3.football.api-sports.io"
API_KEY = os.getenv("RAPIDAPI_KEY", "") # 환경변수 사용

# 리그 정보 (EPL = 39)
LEAGUE_ID = 39
SEASON = 2025

class EPLRealtimeIngestor:
    def __init__(self, mode="real"):
        self.db = EPLDuckDBManager()
        self.mode = mode if API_KEY else "mock"
        if self.mode == "mock":
            print("⚠️ API_KEY missing. Running in MOCK mode for demonstration.")

    def fetch_api(self, endpoint):
        """API 호출 함수"""
        if self.mode == "mock":
            return self._get_mock_data(endpoint)

        conn = http.client.HTTPSConnection(API_HOST)
        headers = {'x-apisports-key': API_KEY}
        
        try:
            conn.request("GET", endpoint, headers=headers)
            res = conn.getresponse()
            data = res.read()
            if res.status != 200:
                print(f"❌ API Error {res.status}")
                return None
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
        finally:
            conn.close()

    def _get_mock_data(self, endpoint):
        """테스트를 위한 Mock 데이터 생성"""
        now = datetime.now()
        if "fixtures" in endpoint and "next" in endpoint:
            return {
                "response": [
                    {
                        "fixture": {"id": 1001, "date": (now + timedelta(hours=2)).isoformat(), "status": {"long": "Not Started"}, "venue": {"name": "Old Trafford"}},
                        "teams": {"home": {"name": "Man United"}, "away": {"name": "Arsenal"}}
                    },
                    {
                        "fixture": {"id": 1002, "date": (now - timedelta(minutes=60)).isoformat(), "status": {"long": "In Progress"}, "venue": {"name": "Anfield"}},
                        "teams": {"home": {"name": "Liverpool"}, "away": {"name": "Chelsea"}}
                    }
                ]
            }
        elif "odds" in endpoint:
            return {
                "response": [{
                    "bookmakers": [{
                        "name": "Bet365",
                        "bets": [{
                            "name": "Match Winner",
                            "values": [
                                {"value": "Home", "odd": "2.10"},
                                {"value": "Draw", "odd": "3.40"},
                                {"value": "Away", "odd": "3.10"}
                            ]
                        }]
                    }]
                }]
            }
        return {"response": []}

    def sync_fixtures(self):
        """다음 10경기를 동기화"""
        print("📡 Syncing Fixtures...")
        data = self.fetch_api(f"/v3/fixtures?league={LEAGUE_ID}&season={SEASON}&next=10")
        if data and data.get('response'):
            rows = []
            for item in data['response']:
                f = item['fixture']
                t = item['teams']
                rows.append({
                    "fixture_id": f['id'],
                    "date": f['date'].replace('T', ' ').split('+')[0],
                    "home_team": t['home']['name'],
                    "away_team": t['away']['name'],
                    "status": f['status']['long'],
                    "venue": f['venue']['name']
                })
            df = pd.DataFrame(rows)
            self.db.insert_fixtures(df)
            return df
        return pd.DataFrame()

    def sync_odds(self, fixture_id):
        """특정 경기의 배당률 동기화"""
        print(f"📡 Syncing Odds for Fixture {fixture_id}...")
        data = self.fetch_api(f"/v3/odds?fixture={fixture_id}")
        if data and data.get('response'):
            # 첫 번째 북메이킹 데이터 사용
            response = data['response'][0]
            for bm in response.get('bookmakers', []):
                if bm['name'] == "Bet365": # 기본적으로 Bet365 사용
                    odds_dict = {"Home": 0, "Draw": 0, "Away": 0}
                    for bet in bm['bets']:
                        if bet['name'] == "Match Winner":
                            for val in bet['values']:
                                odds_dict[val['value']] = float(val['odd'])
                    
                    df = pd.DataFrame([{
                        "fixture_id": fixture_id,
                        "timestamp": datetime.now(),
                        "bookmaker": "Bet365",
                        "home_win_odds": odds_dict['Home'],
                        "draw_odds": odds_dict['Draw'],
                        "away_win_odds": odds_dict['Away']
                    }])
                    self.db.insert_odds(df)
                    return df
        return pd.DataFrame()

    def run_ingestion_loop(self):
        """전체 수집 프로세스 실행"""
        fixtures_df = self.sync_fixtures()
        if not fixtures_df.empty:
            for fid in fixtures_df['fixture_id']:
                self.sync_odds(fid)
                # API Rate Limit 준수 (Mock 제외)
                if self.mode != "mock": time.sleep(1)
        print("✅ Ingestion Complete.")

if __name__ == "__main__":
    ingestor = EPLRealtimeIngestor()
    ingestor.run_ingestion_loop()
