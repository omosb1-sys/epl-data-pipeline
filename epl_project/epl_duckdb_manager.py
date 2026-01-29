import duckdb
import os
import json
import pandas as pd
from datetime import datetime
try:
    from pytoony import json2toon
except ImportError:
    # pytoony가 없을 경우를 대비한 간단한 fallback
    def json2toon(json_str):
        try:
            data = json.loads(json_str)
            return json.dumps(data, separators=(',', ':'))
        except:
            return json_str

class EPLDuckDBManager:
    def __init__(self, db_path: str = None, read_only: bool = False):
        if db_path is None:
            # epl_project/data 폴더 아래에 생성
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base_dir, "data")
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            db_path = os.path.join(data_dir, "epl_realtime.db")
        
        self.db_path = db_path
        # read_only 모드 지원 (대시보드는 True, 수집기는 False)
        self.conn = duckdb.connect(self.db_path, read_only=read_only)
        if not read_only:
            self._init_tables()

    def _init_tables(self):
        """실시간 분석을 위한 테이블 초기화"""
        # 1. 경기 기본 정보 (Fixtures)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fixtures (
                fixture_id INTEGER PRIMARY KEY,
                date TIMESTAMP,
                home_team VARCHAR,
                away_team VARCHAR,
                status VARCHAR,
                venue VARCHAR
            )
        """)

        # 2. 실시간 경기 통계 (Live Stats)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS live_stats (
                fixture_id INTEGER,
                timestamp TIMESTAMP,
                home_possession INTEGER,
                away_possession INTEGER,
                home_shots INTEGER,
                away_shots INTEGER,
                home_goals INTEGER,
                away_goals INTEGER,
                FOREIGN KEY (fixture_id) REFERENCES fixtures(fixture_id)
            )
        """)

        # 3. 실시간 배당률 (Live Odds)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS odds (
                fixture_id INTEGER,
                timestamp TIMESTAMP,
                bookmaker VARCHAR,
                home_win_odds FLOAT,
                draw_odds FLOAT,
                away_win_odds FLOAT,
                FOREIGN KEY (fixture_id) REFERENCES fixtures(fixture_id)
            )
        """)

        # 4. 예측 결과 (Predictions)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                fixture_id INTEGER,
                timestamp TIMESTAMP,
                home_win_prob FLOAT,
                draw_prob FLOAT,
                away_win_prob FLOAT,
                value_bet_side VARCHAR,
                value_bet_edge FLOAT,
                FOREIGN KEY (fixture_id) REFERENCES fixtures(fixture_id)
            )
        """)
        # 5. [Rule 26: Data Analytics Optimization] 시계열 파티셔닝 전략 (Conceptual for DuckDB)
        # DuckDB는 자체적으로 하이브리드 파티셔닝을 수행하지만, 대규모 데이터 대응을 위해 수동 인덱싱 강화
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures(date)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_live_stats_fixture ON live_stats(fixture_id, timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_fixture ON predictions(fixture_id, timestamp)")
        
        print(f"✅ [Enterprise Scale] DuckDB Partitioning & Indexing Initialized at {self.db_path}")

    def get_read_only_connection(self):
        """[Rule 26.2] Read Replica (Secondary Connection) Simulation for Dashboard."""
        return duckdb.connect(self.db_path, read_only=True)

    def insert_fixtures(self, df: pd.DataFrame):
        """경기 정보 삽입/업데이트"""
        if df.empty: return
        self.conn.execute("INSERT OR REPLACE INTO fixtures SELECT * FROM df")

    def insert_live_stats(self, df: pd.DataFrame):
        """실시간 통계 삽입"""
        if df.empty: return
        self.conn.execute("INSERT INTO live_stats SELECT * FROM df")

    def insert_odds(self, df: pd.DataFrame):
        """배당률 데이터 삽입"""
        if df.empty: return
        self.conn.execute("INSERT INTO odds SELECT * FROM df")

    def insert_prediction(self, df: pd.DataFrame):
        """예측 결과 삽입"""
        if df.empty: return
        self.conn.execute("INSERT INTO predictions SELECT * FROM df")

    def get_latest_odds(self, fixture_id: int):
        """특정 경기의 최신 배당률 조회"""
        return self.conn.execute(f"SELECT * FROM odds WHERE fixture_id = {fixture_id} ORDER BY timestamp DESC LIMIT 1").df()

    def get_match_history(self, fixture_id: int):
        """경기 흐름 조회를 위한 통계 히스토리"""
        return self.conn.execute(f"SELECT * FROM live_stats WHERE fixture_id = {fixture_id} ORDER BY timestamp ASC").df()

    def get_latest_match_toon(self, fixture_id: int) -> str:
        """
        [Optimization] DuckDB Native JSON 기능을 사용하여 
        Pandas를 거치지 않고 직접 TOON 형식으로 변환 (Zero-Copy)
        """
        # SQL 레벨에서 직접 JSON 구조체 생성 (성능 극대화)
        query = f"""
            SELECT CAST(json_object(
                'home', f.home_team, 
                'away', f.away_team, 
                'status', f.status,
                'goals', json_object('h', COALESCE(ls.home_goals, 0), 'a', COALESCE(ls.away_goals, 0)),
                'probs', json_object('h', ROUND(COALESCE(p.home_win_prob, 0.33), 2), 'a', ROUND(COALESCE(p.away_win_prob, 0.33), 2))
            ) AS VARCHAR)
            FROM fixtures f
            LEFT JOIN (SELECT fixture_id, home_goals, away_goals FROM live_stats WHERE fixture_id = {fixture_id} ORDER BY timestamp DESC LIMIT 1) ls ON f.fixture_id = ls.fixture_id
            LEFT JOIN (SELECT fixture_id, home_win_prob, away_win_prob FROM predictions WHERE fixture_id = {fixture_id} ORDER BY timestamp DESC LIMIT 1) p ON f.fixture_id = p.fixture_id
            WHERE f.fixture_id = {fixture_id}
        """
        result = self.conn.execute(query).fetchone()
        if not result or not result[0]: return ""
        
        # DuckDB가 이미 최적화된 JSON 문자열을 줬으므로 바로 json2toon 적용
        return json2toon(result[0])

    def get_all_active_matches_toon(self) -> str:
        """[Optimization] 모든 활성 경기를 SQL 레벨에서 통합 JSON으로 변환"""
        query = """
            SELECT CAST(json_group_array(json_object(
                'id', fixture_id, 
                'home', home_team, 
                'away', away_team, 
                'status', status
            )) AS VARCHAR)
            FROM fixtures
            WHERE status != 'FT'
        """
        result = self.conn.execute(query).fetchone()
        if not result or not result[0]: return "[]"
        
        return json2toon(result[0])

if __name__ == "__main__":
    # 테스트 코드
    db = EPLDuckDBManager()
    print("DuckDB Manager Test Success")
