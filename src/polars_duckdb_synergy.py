import polars as pl
import duckdb
import os
import time

# 데이터 경로 설정
RAW_DATA_PATH = 'data/raw/match_info.csv'
DB_PATH = ':memory:' # 인메모리 DB 사용 (빠른 처리)

class BigDataEngine:
    def __init__(self, data_path=RAW_DATA_PATH):
        self.data_path = data_path
        self.con = duckdb.connect(DB_PATH)
        
    def check_data(self):
        if not os.path.exists(self.data_path):
            print(f"⚠️ 데이터 파일이 없습니다: {self.data_path}")
            return False
        return True

    def run_advanced_wrangling(self):
        """
        DuckDB를 사용하여 복잡한 집계와 윈도우 함수 연산을 수행한 후,
        Zero-Copy로 Polars로 변환하여 분석 준비를 마칩니다.
        """
        if not self.check_data(): return

        print("🚀 [Step 1] DuckDB 엔진 가동: 복잡한 SQL 집계 수행")
        start_time = time.time()

        # 시나리오: 
        # 1. 2024시즌 데이터만 필터링
        # 2. 홈/원정 구분 없이 팀별 통합 득점 통계 산출
        # 3. 득점 순위(Rank) 계산 (Window Function)
        
        query = f"""
        WITH home_stats AS (
            SELECT 
                home_team_name_ko as team, 
                SUM(home_score) as goals,
                COUNT(*) as games
            FROM read_csv_auto('{self.data_path}')
            WHERE season_name = 2024
            GROUP BY home_team_name_ko
        ),
        away_stats AS (
            SELECT 
                away_team_name_ko as team, 
                SUM(away_score) as goals,
                COUNT(*) as games
            FROM read_csv_auto('{self.data_path}')
            WHERE season_name = 2024
            GROUP BY away_team_name_ko
        ),
        combined AS (
            SELECT 
                h.team,
                (h.goals + a.goals) as total_goals,
                (h.games + a.games) as total_games
            FROM home_stats h
            JOIN away_stats a ON h.team = a.team
        )
        SELECT 
            team,
            total_goals,
            total_games,
            CAST(total_goals AS FLOAT) / total_games as avg_goals,
            RANK() OVER (ORDER BY total_goals DESC) as goal_rank
        FROM combined
        ORDER BY goal_rank
        """
        
        # DuckDB -> Polars (Zero Copy)
        # .pl() 메서드는 Arrow 포맷을 통해 메모리 복사 없이 데이터를 전달합니다.
        df_pl = self.con.execute(query).pl()
        
        elapsed = time.time() - start_time
        print(f"✅ 처리 완료 ({elapsed:.4f}초 소요)")
        
        return df_pl

    def analyze_results(self, df):
        print("\n📊 [Step 2] Polars 분석 결과 (2024 시즌 득점 랭킹):")
        
        # Polars의 강력한 표현식 사용
        # 예: 경기당 득점이 1.5 이상인 공격적 팀 필터링
        aggressive_teams = df.filter(pl.col("avg_goals") >= 1.5)
        
        print(df)
        
        if not aggressive_teams.is_empty():
            print("\n🔥 [Insight] '닥공' 팀 (평균 1.5골 이상):")
            print(aggressive_teams.select(['team', 'avg_goals']))
        else:
            print("\n❄️ 평균 1.5골 이상을 넣은 팀이 없습니다. (수비적 리그)")

if __name__ == "__main__":
    engine = BigDataEngine()
    result_df = engine.run_advanced_wrangling()
    if result_df is not None:
        engine.analyze_results(result_df)
