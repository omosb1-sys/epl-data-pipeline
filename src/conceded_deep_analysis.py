import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 한글 폰트 설정 (Mac)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def run_analysis():
    conn = sqlite3.connect('data/processed/kleague.db')
    
    print("1. 팀별 평균 실점 및 수비 지표 분석 중...")
    # match_info를 기반으로 팀별 실점 계산
    # 홈팀일 때는 away_score가 실점, 어웨이팀일 때는 home_score가 실점
    query = """
    SELECT 
        team_name_ko,
        COUNT(*) as games_played,
        SUM(case when is_home = 1 then away_score else home_score end) as total_goals_against,
        AVG(case when is_home = 1 then away_score else home_score end) as avg_goals_against
    FROM (
        SELECT home_team_name_ko as team_name_ko, home_score, away_score, 1 as is_home FROM match_info
        UNION ALL
        SELECT away_team_name_ko as team_name_ko, home_score, away_score, 0 as is_home FROM match_info
    )
    GROUP BY team_name_ko
    ORDER BY avg_goals_against ASC
    """
    defense_ranking = pd.read_sql(query, conn)
    
    print("2. 시간대별 실점 패턴 분석 중...")
    # raw_data에서 실점(상대팀의 Goal) 시간대 추출
    # team_id가 실점한 팀이어야 하므로, Goal 이벤트의 반대 팀을 찾아야 함
    # 하지만 Goal 이벤트는 득점한 팀의 team_id를 가짐.
    goal_time_query = """
    SELECT 
        m.game_id,
        r.time_seconds,
        r.period_id,
        r.team_id as scoring_team_id,
        CASE WHEN r.team_id = m.home_team_id THEN m.away_team_name_ko ELSE m.home_team_name_ko END as conceded_team_name_ko
    FROM raw_data r
    JOIN match_info m ON r.game_id = m.game_id
    WHERE r.type_name = 'Goal'
    """
    goals_df = pd.read_sql(goal_time_query, conn)
    
    # 시간대 처리 (전반 0-45, 후반 45-90+)
    def get_match_minute(row):
        m = row['time_seconds'] / 60
        if row['period_id'] == 2:
            return m + 45
        return m
    
    goals_df['match_min'] = goals_df.apply(get_match_minute, axis=1)
    
    # 15분 단위 구간 설정
    bins = [0, 15, 30, 45, 60, 75, 105]
    labels = ['0-15', '15-30', '31-45', '46-60', '61-75', '75+']
    goals_df['time_bin'] = pd.cut(goals_df['match_min'], bins=bins, labels=labels)
    
    time_analysis = goals_df.groupby(['conceded_team_name_ko', 'time_bin'], observed=False).size().unstack(fill_value=0)
    
    # 3. 인사이트 추출 및 시각화
    plt.figure(figsize=(15, 10))
    
    # 그래프 1: 평균 실점 순위
    plt.subplot(2, 1, 1)
    sns.barplot(x='avg_goals_against', y='team_name_ko', data=defense_ranking, palette='coolwarm_r')
    plt.title('K리그 팀별 경기당 평균 실점 (낮을수록 수비 강팀)', fontsize=15)
    plt.xlabel('평균 실점')
    plt.ylabel('팀명')
    
    # 그래프 2: 후반 75분 이후 실점 비중
    plt.subplot(2, 1, 2)
    late_goals = time_analysis['75+'] / time_analysis.sum(axis=1) * 100
    late_goals = late_goals.sort_values(ascending=False).reset_index()
    late_goals.columns = ['team_name_ko', 'late_goal_pct']
    
    sns.barplot(x='late_goal_pct', y='team_name_ko', data=late_goals, palette='Reds')
    plt.title('전체 실점 중 후반 75분 이후 실점 비중 (%) - 집중력 지표', fontsize=15)
    plt.xlabel('75분 이후 실점 비중 (%)')
    plt.ylabel('팀명')
    
    plt.tight_layout()
    plt.savefig('reports/figures/reports/figures/conceded_analysis_report.png', dpi=300)
    
    # 텍스트 인사이트 출력
    print("\n" + "="*50)
    print("📊 K리그 실점 데이터 분석 인사이트")
    print("="*50)
    
    best_def = defense_ranking.iloc[0]
    worst_def = defense_ranking.iloc[-1]
    print(f"✅ 최강 방패: {best_def['team_name_ko']} (경기당 {best_def['avg_goals_against']:.2f}실점)")
    print(f"⚠️ 수비 보완 필요: {worst_def['team_name_ko']} (경기당 {worst_def['avg_goals_against']:.2f}실점)")
    
    most_late = late_goals.iloc[0]
    print(f"⏰ 후반 막판 주의보: {most_late['team_name_ko']} (실점의 {most_late['late_goal_pct']:.1f}%가 75분 이후 발생)")
    print("   -> 경기가 끝날 때까지의 집중력 유지가 핵심 과제로 보입니다.")
    
    print("\n🖼️ 상세 분석 결과가 'conceded_analysis_report.png'로 저장되었습니다.")
    
    conn.close()

if __name__ == "__main__":
    run_analysis()
