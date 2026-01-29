import duckdb
import polars as pl
import os
from typing import Optional

class BigDataEngine:
    """
    K-League & EPL 대용량 데이터 전처리 엔진 (Refactored)
    DuckDB의 SQL 파워와 Polars의 고속 연산을 결합합니다.
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        print(f"📊 BigDataEngine 가동: {data_path}")

    def run_advanced_wrangling(self) -> Optional[pl.DataFrame]:
        """고속 데이터 랭글링 실행"""
        if not os.path.exists(self.data_path):
            return None
        
        # 1. DuckDB로 대용량 데이터 로드
        con = duckdb.connect(database=':memory:')
        # CSV를 전처리하여 읽기
        df_pl = con.execute(f"SELECT * FROM read_csv_auto('{self.data_path}')").pl()
        
        # 2. Polars로 고속 집계
        # 팀별 평균 득점 계산
        df_result = df_pl.lazy().group_by("home_team_name_ko").agg([
            pl.col("home_score").mean().alias("avg_goals")
        ]).rename({"home_team_name_ko": "team"}).collect()
        
        return df_result
