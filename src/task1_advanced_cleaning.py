import pandas as pd
import numpy as np
import os
import duckdb # SQL 기반의 고성능 분석 라이브러리 (대용량 csv 로드에 최적화)

def task1_advanced_preprocessing(raw_data_path, match_info_path):
    """
    원본 데이터를 불러와 전처리 및 피처 엔지니어링을 수행하는 메인 함수
    """
    print(f"--- Task 1: Advanced Preprocessing & Engineering (DuckDB Loaded) ---")
    
    # 1. 데이터 로드 (DuckDB를 사용하여 초고속 로드 및 Pandas 변환)
    try:
        # DuckDB 인메모리 연결
        con = duckdb.connect(database=':memory:')
        
        # SQL을 사용하여 CSV 로드 (Pandas read_csv보다 대용량 처리에 유리함)
        raw_data = con.execute(f"SELECT * FROM read_csv_auto('{raw_data_path}')").df()
        match_info = con.execute(f"SELECT * FROM read_csv_auto('{match_info_path}')").df()
        
        print(f"✓ DuckDB 로드 성공: raw_data {raw_data.shape}, match_info {match_info.shape}")
    except Exception as e:
        print(f"❌ 데이터 로드 중 오류 발생: {e}")
        return None

    # 2. 기초 전처리 (날짜 형식 변환 및 결측치 처리)
    match_info['game_date'] = pd.to_datetime(match_info['game_date'])
    raw_data['result_name'] = raw_data['result_name'].fillna('Unknown')
    
    # 3. 로그 데이터를 팀별 경기 통계로 집계 (Feature Aggregation)
    print("📊 이벤트 데이터를 기반으로 경기별 통계 집계 중...")
    
    # 각 경기(game_id), 팀(team_id)별로 그룹화하여 주요 지표 계산
    game_team_stats = raw_data.groupby(['game_id', 'team_id', 'team_name_ko']).apply(lambda x: pd.Series({
        'total_passes': (x['type_name'] == 'Pass').sum(),  # 전체 패스 횟수
        'successful_passes': ((x['type_name'] == 'Pass') & (x['result_name'] == 'Successful')).sum(), # 성공한 패스
        'total_shots': (x['type_name'].isin(['Shot', 'Goal', 'Shot_Freekick'])).sum(), # 전체 슈팅
        'goals': (x['type_name'] == 'Goal').sum() + ((x['type_name'] == 'Shot') & (x['result_name'] == 'Goal')).sum(), # 득점
        'tackles': (x['type_name'] == 'Tackle').sum(), # 태클 성공
        'interceptions': (x['type_name'] == 'Interception').sum(), # 가로채기
        'fouls': (x['type_name'].str.contains('Foul', na=False)).sum(), # 반칙 횟수
        'attack_zone_actions': (x['start_x'] > 60).sum() # 공격 진영(상대방 진영 40%)에서의 활동량
    }), include_groups=False).reset_index()
    
    # 패스 성공률 계산 (0으로 나누기 방지)
    game_team_stats['pass_success_rate'] = (game_team_stats['successful_passes'] / game_team_stats['total_passes'].replace(0, 1)) * 100
    
    # 4. 경기 정보 결합 (홈/어웨이 정보 등)
    game_team_stats = game_team_stats.merge(
        match_info[['game_id', 'home_team_id', 'away_team_id', 'game_date']],
        on='game_id'
    )
    
    # 상대팀의 득점(실점) 계산을위한 셀프 조인
    temp_goals = game_team_stats[['game_id', 'team_id', 'goals']].rename(columns={'team_id': 'opp_id', 'goals': 'goals_against'})
    game_team_stats = game_team_stats.merge(temp_goals, on='game_id')
    game_team_stats = game_team_stats[game_team_stats['team_id'] != game_team_stats['opp_id']]
    
    # 홈 경기 여부 확인 (1: 홈, 0: 어웨이)
    game_team_stats['is_home'] = (game_team_stats['team_id'] == game_team_stats['home_team_id']).astype(int)
    
    # 타겟 변수 설정 (승리 또는 무승부 여부: 1, 패배: 0)
    game_team_stats['is_win'] = (game_team_stats['goals'] >= game_team_stats['goals_against']).astype(int)
    
    # 5. 고급 피처 엔지니어링 수행
    print("🔧 고급 파생 변수 생성 중...")
    
    # 슈팅 당 실효성 (득점 / 전체 슈팅)
    game_team_stats['shot_efficiency'] = np.where(
        game_team_stats['total_shots'] > 0, 
        game_team_stats['goals'] / game_team_stats['total_shots'], 
        0
    )
    # 수비 압박 강도 (태클 + 가로채기)
    game_team_stats['defensive_pressure'] = game_team_stats['tackles'] + game_team_stats['interceptions']
    
    # 최근 경기 흐름 (최근 3경기 평균 승률 및 패스 성공률)
    game_team_stats = game_team_stats.sort_values(['team_id', 'game_date'])
    # 이전 경기들의 기록을 shift하여 현재 분석 모델이 미래 정보를 미리 알지 못하도록(Data Leakage 방지) 처리
    game_team_stats['rolling_win_rate'] = game_team_stats.groupby('team_id')['is_win'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    ).fillna(0)
    
    game_team_stats['rolling_pass_rate'] = game_team_stats.groupby('team_id')['pass_success_rate'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    ).fillna(game_team_stats['pass_success_rate'].mean())

    # 6. 정제된 데이터셋 저장
    output_path = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/processed/processed_ml_data.csv"
    game_team_stats.to_csv(output_path, index=False)
    print(f"✓ 정제된 데이터가 저장되었습니다: {output_path}")
    
    return game_team_stats

if __name__ == "__main__":
    RAW_DATA = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/raw/raw_data.csv"
    MATCH_INFO = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/raw/match_info.csv"
    task1_advanced_preprocessing(RAW_DATA, MATCH_INFO)

if __name__ == "__main__":
    RAW_DATA = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/raw/raw_data.csv"
    MATCH_INFO = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/raw/match_info.csv"
    task1_advanced_preprocessing(RAW_DATA, MATCH_INFO)
