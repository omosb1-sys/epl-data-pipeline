import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 한글 폰트 설정 (Mac)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def run_pure_pandas_analysis():
    print("CSV 파일 로딩 중... (데이터가 커서 시간이 조금 걸릴 수 있습니다)")
    raw_data = pd.read_csv('data/raw/raw_data.csv')
    match_info = pd.read_csv('data/raw/match_info.csv')
    
    # ---------------------------------------------------------
    # [분석 1] 팀별 평균 실점 및 수비 지표 분석 (SQL의 SELECT + UNION ALL 역할)
    # ---------------------------------------------------------
    print("1. 팀별 평균 실점 분석 중 (Pandas 방식)...")
    
    # SQL의 UNION ALL처럼 홈팀 입장과 어웨이팀 입장의 데이터를 각각 만듭니다.
    home_side = match_info[['home_team_name_ko', 'home_score', 'away_score']].rename(
        columns={'home_team_name_ko': 'team_name_ko', 'home_score': 'our_score', 'away_score': 'goals_against'}
    )
    away_side = match_info[['away_team_name_ko', 'home_score', 'away_score']].rename(
        columns={'away_team_name_ko': 'team_name_ko', 'home_score': 'goals_against', 'away_score': 'our_score'}
    )
    
    # 두 데이터를 세로로 합칩니다 (SQL의 UNION ALL)
    team_match_results = pd.concat([home_side, away_side], axis=0)
    
    # 팀별로 그룹화하여 평균 계산 (SQL의 GROUP BY + AVG)
    defense_ranking = team_match_results.groupby('team_name_ko')['goals_against'].agg(['count', 'sum', 'mean']).reset_index()
    defense_ranking.columns = ['team_name_ko', 'games_played', 'total_goals_against', 'avg_goals_against']
    defense_ranking = defense_ranking.sort_values('avg_goals_against', ascending=True)
    
    # ---------------------------------------------------------
    # [분석 2] 시간대별 실점 패턴 분석 (SQL의 JOIN + CASE WHEN 역할)
    # ---------------------------------------------------------
    print("2. 시간대별 실점 패턴 분석 중 (Pandas 방식)...")
    
    # 득점 상황만 필터링 (SQL의 WHERE type_name = 'Goal')
    goals_only = raw_data[raw_data['type_name'] == 'Goal'].copy()
    
    # 경기 정보와 합치기 (SQL의 JOIN)
    goals_with_info = goals_only.merge(match_info[['game_id', 'home_team_id', 'home_team_name_ko', 'away_team_name_ko']], on='game_id', how='left')
    
    # 실점한 팀 찾아내기 (SQL의 CASE r.team_id = m.home_team_id THEN ...)
    def identify_conceded_team(row):
        if row['team_id'] == row['home_team_id']:
            return row['away_team_name_ko'] # 홈팀이 골 넣었으니 어웨이팀이 실점
        else:
            return row['home_team_name_ko'] # 어웨이팀이 골 넣었으니 홈팀이 실점
            
    goals_with_info['conceded_team_name_ko'] = goals_with_info.apply(identify_conceded_team, axis=1)
    
    # 시간(분) 계산
    goals_with_info['match_min'] = goals_with_info.apply(
        lambda row: (row['time_seconds'] / 60) + 45 if row['period_id'] == 2 else (row['time_seconds'] / 60), 
        axis=1
    )
    
    # 구간 나누기 (Cut)
    bins = [0, 15, 30, 45, 60, 75, 105]
    labels = ['0-15', '15-30', '31-45', '46-60', '61-75', '75+']
    goals_with_info['time_bin'] = pd.cut(goals_with_info['match_min'], bins=bins, labels=labels)
    
    # 피벗 테이블 생성 (SQL의 GROUP BY + unstack)
    time_analysis = goals_with_info.groupby(['conceded_team_name_ko', 'time_bin'], observed=False).size().unstack(fill_value=0)
    
    # ---------------------------------------------------------
    # 시각화 (동일)
    # ---------------------------------------------------------
    plt.figure(figsize=(15, 10))
    
    # 그래프 1: 평균 실점
    plt.subplot(2, 1, 1)
    sns.barplot(x='avg_goals_against', y='team_name_ko', data=defense_ranking, palette='coolwarm_r')
    plt.title('[Pandas 분석] K리그 팀별 경기당 평균 실점', fontsize=15)
    
    # 그래프 2: 후반 75분 이후 실점 비중
    plt.subplot(2, 1, 2)
    late_goals = (time_analysis['75+'] / time_analysis.sum(axis=1) * 100).sort_values(ascending=False).reset_index()
    late_goals.columns = ['team_name_ko', 'late_goal_pct']
    
    sns.barplot(x='late_goal_pct', y='team_name_ko', data=late_goals, palette='Reds')
    plt.title('[Pandas 분석] 전체 실점 중 75분 이후 비중 (%)', fontsize=15)
    
    plt.tight_layout()
    plt.savefig('reports/figures/reports/figures/conceded_analysis_pandas.png', dpi=300)
    print("\n✓ 'conceded_analysis_pandas.png' 저장 완료!")
    
    print("\n" + "="*50)
    print("📊 Pandas 방식 결과 요약")
    print("="*50)
    print(f"최강 방패: {defense_ranking.iloc[0]['team_name_ko']} ({defense_ranking.iloc[0]['avg_goals_against']:.2f}실점)")
    print(f"집중력 확인 필요: {late_goals.iloc[0]['team_name_ko']} (75분 후 실점 비중 {late_goals.iloc[0]['late_goal_pct']:.1f}%)")

if __name__ == "__main__":
    run_pure_pandas_analysis()
