import duckdb
import polars as pl
import os
from typing import Tuple, List, Dict

# --- TOON (Token-Oriented Object Notation) Utility ---
def to_toon(data: List[Dict], label: str = "DATA") -> str:
    """
    JSON의 불필요한 토큰을 제거한 LLM 최적화 포맷입니다.
    """
    lines = [f"▼ {label}"]
    for item in data:
        # 특수문자를 최소화하고 바(|)로 구분하여 토큰 효율 극대화
        fields = [f"{k}: {v}" for k, v in item.items() if v is not None]
        lines.append(f"  ● " + " | ".join(fields))
    return "\n".join(lines)

# --- 30년 차 시니어 분석가의 고성능 파이프라인 ---

def duckdb_sql_layer(parquet_path: str) -> pl.DataFrame:
    """
    Parquet 파일을 DuckDB로 직접 쿼리합니다.
    Parquet의 컬럼 기반 저장 방식 덕분에 필요한 컬럼만 선택적으로 읽어와 속도가 압도적입니다.
    """
    print(f"🪄 [Layer 1: DuckDB] Parquet Engine 가동 - {parquet_path} 직독 수행...")
    
    query = f"""
    SELECT 
        team, 
        game_id, 
        opponent,
        goals_for, 
        goals_against, 
        result,
        points,
        CAST(SUBSTR(CAST(game_id AS VARCHAR), 1, 4) AS INTEGER) as season
    FROM read_parquet('{parquet_path}')
    WHERE points IS NOT NULL
    ORDER BY team, game_id
    """
    
    # DuckDB 결과를 Polars로 변환 (Arrow를 통해 제로카피로 연결되어 매우 빠름)
    df_pl = duckdb.query(query).pl()
    return df_pl

def polars_kinetic_layer(df: pl.DataFrame) -> pl.DataFrame:
    """
    Polars의 지연 연산(Lazy Evaluation)을 사용하여 복잡한 시계열 특징을 생성합니다.
    파이썬의 유연성과 Rust의 속도를 결합합니다.
    """
    print("⚡ [Layer 2: Polars] Kinetic Engine 가동 - 복잡한 특징 엔지니어링 수행...")
    
    # 지연 연산을 위해 lazy() 모드 진입
    df_result = df.lazy().with_columns([
        # 1. 최근 5경기 이동 평균 득점 (팀별 그룹화)
        pl.col("goals_for").rolling_mean(window_size=5, min_periods=1).over("team").alias("rolling_avg_goals"),
        
        # 2. 누적 승점 (시즌별, 팀별 그룹화)
        pl.col("points").cum_sum().over(["season", "team"]).alias("cumulative_points"),
        
        # 3. 모멘텀 지표: 최근 3경기 승점 합계
        pl.col("points").rolling_sum(window_size=3, min_periods=1).over("team").alias("momentum_score"),
        
        # 4. 결과 이산화 (Win=1, else=0)
        pl.when(pl.col("result") == "Win").then(1).otherwise(0).alias("is_win")
    ]).with_columns([
        # 5. 최근 5경기 승률 (변환된 is_win 사용)
        pl.col("is_win").rolling_mean(window_size=5, min_periods=1).over("team").alias("recent_win_rate")
    ]).collect() # 최종 단계에서만 실제 연산 수행
    
    return df_result

def senior_analyst_report(df: pl.DataFrame):
    """
    30년 차 시니어 분석가의 관점에서 데이터를 해석합니다.
    """
    print("\n" + "="*50)
    print("📊 [Layer 3: Dynamic] Senior Analyst Insight Report")
    print("="*50)
    
    # 상위 모멘텀 팀 추출
    top_momentum = df.sort("momentum_score", descending=True).head(5)
    
    print("\n🚀 [TOON Format Insight: 최적화된 데이터 전송]")
    # Polars DataFrame을 딕셔너리 리스트로 변환 후 TOON으로 변환
    toon_output = to_toon(top_momentum.to_dicts(), label="TOP_MOMENTUM_TEAMS")
    print(toon_output)
    
    print("\n💡 [시니어의 한마디]")
    print("DuckDB의 SQL로 필터링한 후, Polars의 벡터 연산으로 특징을 뽑아내는 방식은")
    print("기존 Pandas 대비 약 10~50배 이상의 성능 이득을 줍니다. 버그 없는 파이프라인의 핵심은")
    print("데이터를 '루프(Loop)'로 돌리지 않고 '벡터(Vector)'로 처리하는 데 있습니다.")
    print("="*50)

def main():
    parquet_path = 'data/processed/team_match_results.parquet'
    
    # 1. DuckDB Layer (SQL on Parquet)
    df_duck = duckdb_sql_layer(parquet_path)
    
    # 2. Polars Layer (Feature Engineering)
    df_final = polars_kinetic_layer(df_duck)
    
    # 3. Report
    senior_analyst_report(df_final)

if __name__ == "__main__":
    main()
