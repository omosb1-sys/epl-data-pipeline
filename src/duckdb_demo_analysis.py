import duckdb
import time

def run_duckdb_demo():
    print("🚀 DuckDB 초고속 분석 시작...")
    
    # 1. CSV 파일 직접 쿼리 (가장 강력한 기능!)
    # 별도의 DB 생성 없이 바로 CSV를 읽어서 분석합니다.
    start_time = time.time()
    
    print("\n[분석 1] 포지션별 패스 성공률 분석 (CSV 직접 쿼리)")
    # SQL 문 내에 파일 경로를 직접 넣습니다.
    query1 = """
    SELECT 
        position_name,
        COUNT(*) as total_actions,
        AVG(CASE WHEN result_name = 'Successful' THEN 100.0 ELSE 0.0 END) as pass_accuracy
    FROM 'data/raw/raw_data.csv'
    WHERE type_name IN ('Pass', 'Cross')
    GROUP BY position_name
    HAVING total_actions > 100
    ORDER BY pass_accuracy DESC
    """
    
    result1 = duckdb.query(query1).df()
    print(result1)
    print(f"⏱️ 소요 시간: {time.time() - start_time:.2f}초")

    # 2. 복합 쿼리 (JOIN 연습)
    # 두 개의 CSV 파일을 실시간으로 JOIN 합니다.
    print("\n[분석 2] 팀별 경기당 평균 공격 지역 진입 횟수 (JOIN 분석)")
    start_time = time.time()
    
    query2 = """
    SELECT 
        m.home_team_name_ko as team_name,
        AVG(attack_count) as avg_attack_actions
    FROM 'data/raw/match_info.csv' m
    JOIN (
        SELECT game_id, team_id, COUNT(*) as attack_count
        FROM 'data/raw/raw_data.csv'
        WHERE start_x > 70
        GROUP BY game_id, team_id
    ) r ON m.game_id = r.game_id AND m.home_team_id = r.team_id
    GROUP BY m.home_team_name_ko
    ORDER BY avg_attack_actions DESC
    LIMIT 5
    """
    
    result2 = duckdb.query(query2).df()
    print(result2)
    print(f"⏱️ 소요 시간: {time.time() - start_time:.2f}초")

    print("\n💡 DuckDB의 특징:")
    print("1. 'data/raw/raw_data.csv'를 별도의 DB로 변환(Import)하지 않고 SQL로 바로 읽었습니다.")
    print("2. 90MB가 넘는 대용량 파일임에도 인덱스 없이 매우 빠른 속도로 처리되었습니다.")
    print("3. 결과가 바로 Pandas DataFrame으로 반환되어 시각화하기 매우 편리합니다.")

if __name__ == "__main__":
    run_duckdb_demo()
