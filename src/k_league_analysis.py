"""
K리그 2024 시즌 종합 데이터 분석
=================================
전처리, EDA, 파생컬럼, 통계분석, 머신러닝, 인사이트 추출

실행 방법: python3 k_league_analysis.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# 일부 환경(특히 샌드박스/제한된 런타임)에서 BLAS/OpenMP 다중 스레드가
# 드물게 비정상 종료(세그폴트)를 유발할 수 있어 기본 스레드를 1로 제한합니다.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# matplotlib이 ~/.matplotlib 등에 캐시를 쓰려다 실패하면 프로세스가 종료되는 경우가 있어
# 실행 폴더(프로젝트 내부)에 캐시/설정을 두고, GUI 없이 파일로만 저장하도록 백엔드를 고정합니다.
PROJECT_DIR = Path(__file__).resolve().parent
MPLCONFIGDIR = PROJECT_DIR / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")  # headless/권한 제한 환경에서도 안정적으로 저장
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings("ignore")

# 한글 폰트 설정 (Mac) - 폰트가 없으면 기본 폰트로 fallback
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("K리그 2024 시즌 종합 데이터 분석")
print("=" * 60)

# =============================================================================
# 1. 데이터 로드 및 전처리 (Data Preprocessing)
# =============================================================================
print("\n[1] 데이터 로드 및 전처리")
print("-" * 40)

# 데이터 로드 (실행 위치가 달라도 동작하도록 스크립트 기준 경로 사용)
raw_path = PROJECT_DIR / "data/raw/raw_data.csv"
match_path = PROJECT_DIR / "data/raw/match_info.csv"
raw_data = pd.read_csv(raw_path, encoding="utf-8")
match_info = pd.read_csv(match_path, encoding="utf-8")

print(f"raw_data 크기: {raw_data.shape}")
print(f"match_info 크기: {match_info.shape}")

# 기본 정보 확인
print("\n[1.1] 결측치 현황")
print("raw_data 결측치:")
missing_raw = raw_data.isnull().sum()
print(missing_raw[missing_raw > 0])

print("\nmatch_info 결측치:")
missing_match = match_info.isnull().sum()
print(missing_match[missing_match > 0])

# 데이터 타입 변환
print("\n[1.2] 데이터 타입 변환")
match_info['game_date'] = pd.to_datetime(match_info['game_date'])
print("game_date를 datetime으로 변환 완료")

# result_name 결측치 처리 (빈 값은 'Unknown'으로)
raw_data['result_name'] = raw_data['result_name'].fillna('Unknown')

# 데이터 병합
print("\n[1.3] 데이터 병합")
df = raw_data.merge(match_info, on='game_id', how='left')
print(f"병합된 데이터 크기: {df.shape}")

# =============================================================================
# 2. 파생 컬럼 생성 (Feature Engineering)
# =============================================================================
print("\n[2] 파생 컬럼 생성")
print("-" * 40)

# 2.1 패스 거리 계산
df['pass_distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
print("✓ pass_distance (패스 거리) 생성")

# 2.2 패스 방향 분류
def classify_pass_direction(dx, dy, type_name):
    """패스 방향 분류: 전진/후방/횡패스"""
    if type_name not in ['Pass', 'Pass_Freekick', 'Pass_Corner', 'Cross', 'Throw-In']:
        return 'Not Applicable'
    if dx > 5:
        return '전진 패스'
    elif dx < -5:
        return '후방 패스'
    else:
        return '횡패스'

df['pass_direction'] = df.apply(
    lambda x: classify_pass_direction(x['dx'], x['dy'], x['type_name']), axis=1
)
print("✓ pass_direction (패스 방향) 생성")

# 2.3 피치 구역 분류 (105m 기준)
def classify_field_zone(x):
    """피치를 3등분하여 구역 분류"""
    if pd.isna(x):
        return 'Unknown'
    if x < 35:
        return '수비 1/3'
    elif x < 70:
        return '중앙 1/3'
    else:
        return '공격 1/3'

# K리그 경기장 표준 길이인 105m를 35m 단위로 3등분하여 수비, 중앙, 공격 구역으로 분류합니다.
df['field_zone'] = df['start_x'].apply(classify_field_zone)
print("✓ field_zone (피치 구역) 생성")

# 2.4 이벤트 성공 여부 이진화
df['is_successful'] = df['result_name'].apply(
    lambda x: 1 if x == 'Successful' else (0 if x == 'Unsuccessful' else np.nan)
)
print("✓ is_successful (성공 여부) 생성")

# 2.5 경기 시간대 구분
def classify_time_period(seconds, period_id):
    """경기 시간대 분류"""
    if pd.isna(seconds):
        return 'Unknown'
    minutes = seconds / 60
    if period_id == 1:
        if minutes < 15:
            return '전반 0-15분'
        elif minutes < 30:
            return '전반 15-30분'
        else:
            return '전반 30-45분+'
    else:
        if minutes < 15:
            return '후반 0-15분'
        elif minutes < 30:
            return '후반 15-30분'
        else:
            return '후반 30-45분+'

df['time_period'] = df.apply(
    lambda x: classify_time_period(x['time_seconds'], x['period_id']), axis=1
)
print("✓ time_period (경기 시간대) 생성")

# 2.6 경기 결과 생성
def get_match_result(row):
    """경기 결과 산출 (해당 팀 기준)"""
    if row['team_id'] == row['home_team_id']:
        if row['home_score'] > row['away_score']:
            return '승리'
        elif row['home_score'] < row['away_score']:
            return '패배'
        else:
            return '무승부'
    else:
        if row['away_score'] > row['home_score']:
            return '승리'
        elif row['away_score'] < row['home_score']:
            return '패배'
        else:
            return '무승부'

df['match_result'] = df.apply(get_match_result, axis=1)
print("✓ match_result (경기 결과) 생성")

print(f"\n파생 컬럼 생성 완료. 현재 컬럼 수: {len(df.columns)}")


# 2.7 고급 비율 지표 생성 (Advanced Metrics)
print("\n[2.7] 고급 비율 지표 생성 (Team Level)")

# (1) 유효 슈팅 비율 (Shooting Accuracy)
# 팀별로 슈팅 관련 데이터 집계
shot_stats = df.groupby('team_name_ko').apply(
    lambda x: pd.Series({
        'total_shots': len(x[x['type_name'].isin(['Shot', 'Shot_Freekick', 'Penalty Kick'])]),
        'goals': len(x[x['type_name'] == 'Goal'])
    })
).astype(int)

# 득점 효율성 (Conversion Rate) 계산: 골 / 전체 슈팅
shot_stats['conversion_rate'] = (shot_stats['goals'] / shot_stats['total_shots'] * 100).fillna(0).round(1)
print(f"✓ 득점 효율성(Conversion Rate) 계산 완료")
print(shot_stats.sort_values('conversion_rate', ascending=False).head(3))


# (2) 전진 패스 비율 (Aggressive Pass Ratio) - 숫자 조건 직접 계산 방식
print("\n[팀별 전진 패스 비율 - 최종 해결 버전]")

# 패스 이벤트 타입 정의
pass_types = ['Pass', 'Pass_Freekick', 'Pass_Corner', 'Cross']

# 그룹화하여 직접 숫자로 계산 (문자열 매칭을 거치지 않음)
pass_stats_adv = df[df['type_name'].isin(pass_types)].groupby('team_name_ko').apply(
    lambda x: pd.Series({
        'total_passes': len(x),
        # dx가 5보다 큰 것이 '전진 패스'의 정의이므로, dx 숫자를 직접 체크!
        'forward_passes': (x['dx'] > 5).sum()
    })
).astype(int)

# 비율 계산
pass_stats_adv['forward_pass_ratio'] = (
    pass_stats_adv['forward_passes'] / pass_stats_adv['total_passes'] * 100
).fillna(0).round(1)

print(pass_stats_adv.sort_values('forward_pass_ratio', ascending=False).head(5))


# (3) 홈/어웨이 승률 비교 (Home/Away Strength)
# 이미 match_result가 있으므로 이를 활용
result_stats = df.groupby(['team_name_ko', 'game_id']).first() # 경기당 1줄로 압축

def calc_win_ratio(data):
    if len(data) == 0: return 0
    return (data['match_result'] == '승리').sum() / len(data) * 100

home_win_rate = result_stats[result_stats['team_id'] == result_stats['home_team_id']].groupby('team_name_ko').apply(calc_win_ratio)
away_win_rate = result_stats[result_stats['team_id'] != result_stats['home_team_id']].groupby('team_name_ko').apply(calc_win_ratio)

travel_strength = pd.DataFrame({
    'home_win_rate': home_win_rate,
    'away_win_rate': away_win_rate
}).fillna(0)

# 원정 강세 지표: 어웨이 승률 / 홈 승률 (높을수록 원정 깡패)
travel_strength['travel_strength_index'] = (travel_strength['away_win_rate'] / travel_strength['home_win_rate']).fillna(0).round(2)
print(f"\n✓ 원정 강세 지표(Travel Strength Index) 계산 완료")
print(travel_strength.sort_values('travel_strength_index', ascending=False).head(3))

    


# =============================================================================
# 3. 탐색적 데이터 분석 (EDA)
# =============================================================================
print("\n[3] 탐색적 데이터 분석 (EDA)")
print("-" * 40)

# 3.1 기초 통계량
print("\n[3.1] 기초 통계량")
print(f"총 경기 수: {df['game_id'].nunique()}")
print(f"총 팀 수: {df['team_name_ko'].nunique()}")
print(f"총 선수 수: {df['player_name_ko'].nunique()}")
print(f"총 이벤트 수: {len(df):,}")

# 3.2 이벤트 타입별 분포
print("\n[3.2] 이벤트 타입별 분포 (상위 15개)")
event_counts = df['type_name'].value_counts()
print(event_counts.head(15))

# 시각화 저장
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 이벤트 타입 분포
ax1 = axes[0, 0]
event_counts.head(15).plot(kind='barh', ax=ax1, color='steelblue')
ax1.set_title('이벤트 타입별 분포 (상위 15개)', fontsize=14)
ax1.set_xlabel('빈도수')
ax1.invert_yaxis()

# 팀별 이벤트 수
ax2 = axes[0, 1]
team_events = df['team_name_ko'].value_counts()
team_events.plot(kind='barh', ax=ax2, color='coral')
ax2.set_title('팀별 총 이벤트 수', fontsize=14)
ax2.set_xlabel('이벤트 수')
ax2.invert_yaxis()

# 시간대별 이벤트 분포
ax3 = axes[1, 0]
time_order = ['전반 0-15분', '전반 15-30분', '전반 30-45분+', 
              '후반 0-15분', '후반 15-30분', '후반 30-45분+']
time_counts = df['time_period'].value_counts().reindex(time_order)
time_counts.plot(kind='bar', ax=ax3, color='mediumseagreen')
ax3.set_title('시간대별 이벤트 분포', fontsize=14)
ax3.set_ylabel('이벤트 수')
ax3.tick_params(axis='x', rotation=45)

# 피치 구역별 이벤트 분포
ax4 = axes[1, 1]
zone_order = ['수비 1/3', '중앙 1/3', '공격 1/3']
zone_counts = df['field_zone'].value_counts().reindex(zone_order)
zone_counts.plot(kind='bar', ax=ax4, color='mediumpurple')
ax4.set_title('피치 구역별 이벤트 분포', fontsize=14)
ax4.set_ylabel('이벤트 수')
ax4.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('reports/figures/reports/figures/eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ EDA 시각화 저장: eda_overview.png")

# 3.3 패스 분석
print("\n[3.3] 패스 분석")
pass_data = df[df['type_name'].isin(['Pass', 'Pass_Freekick', 'Pass_Corner', 'Cross'])]
print(f"총 패스 수: {len(pass_data):,}")

pass_success = pass_data[pass_data['result_name'] == 'Successful']
print(f"성공 패스 수: {len(pass_success):,}")
print(f"전체 패스 성공률: {len(pass_success)/len(pass_data)*100:.1f}%")

# 팀별 패스 성공률
print("\n[팀별 패스 성공률]")
team_pass_stats = pass_data.groupby('team_name_ko').agg({
    'game_id': 'count',
    'is_successful': 'sum'
}).rename(columns={'game_id': '총 패스', 'is_successful': '성공 패스'})
team_pass_stats['성공률(%)'] = (team_pass_stats['성공 패스'] / team_pass_stats['총 패스'] * 100).round(1)
team_pass_stats = team_pass_stats.sort_values('성공률(%)', ascending=False)
print(team_pass_stats)


print("\n[3.3.1] 시간대별 패스 성공률")
# 시간대별로 그룹화하여 성공률 계산
time_pass_stats = pass_data.groupby('time_period').apply(
    lambda x: (x['result_name'] == 'Successful').sum() / len(x) * 100
).reindex(time_order) # 아까 만든 순서대로 정리

print(time_pass_stats.round(1))


# 패스 방향 분석
print("\n[패스 방향별 분포]")
pass_direction_counts = df[df['pass_direction'] != 'Not Applicable']['pass_direction'].value_counts()
print(pass_direction_counts)

# 3.4 슈팅/골 분석
print("\n[3.4] 슈팅/골 분석")
shot_events = ['Shot', 'Shot_Freekick', 'Penalty Kick']
shot_data = df[df['type_name'].isin(shot_events)]
goal_data = df[df['type_name'] == 'Goal']
print(f"총 슈팅 수: {len(shot_data):,}")
print(f"총 골 수: {len(goal_data):,}")
print(f"슈팅 당 골 비율: {len(goal_data)/len(shot_data)*100:.1f}%" if len(shot_data) > 0 else "N/A")

# 팀별 슈팅/골 통계
print("\n[팀별 슈팅/골 통계]")
team_shot_stats = df.groupby('team_name_ko').apply(
    lambda x: pd.Series({
        '슈팅': len(x[x['type_name'].isin(shot_events)]),
        '골': len(x[x['type_name'] == 'Goal'])
    })
).astype(int)
team_shot_stats['득점 효율(%)'] = (team_shot_stats['골'] / team_shot_stats['슈팅'] * 100).round(1)
team_shot_stats = team_shot_stats.sort_values('골', ascending=False)
print(team_shot_stats)

# 슈팅 히트맵
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 슈팅 위치 산점도
ax1 = axes[0]
shot_positions = shot_data[['start_x', 'start_y']].dropna()
ax1.scatter(shot_positions['start_x'], shot_positions['start_y'], 
           alpha=0.3, c='red', s=10)
ax1.set_xlim(0, 105)
ax1.set_ylim(0, 68)
ax1.set_title('슈팅 위치 분포', fontsize=14)
ax1.set_xlabel('X 좌표 (공격 방향 →)')
ax1.set_ylabel('Y 좌표')
# 페널티 박스 표시
ax1.axvline(x=88.5, color='green', linestyle='--', alpha=0.5)
ax1.text(89, 5, '페널티 박스', fontsize=9, color='green')

# 골 위치
ax2 = axes[1]
goal_positions = goal_data[['start_x', 'start_y']].dropna()
ax2.scatter(goal_positions['start_x'], goal_positions['start_y'], 
           alpha=0.6, c='gold', s=50, edgecolors='black')
ax2.set_xlim(0, 105)
ax2.set_ylim(0, 68)
ax2.set_title('골 위치 분포', fontsize=14)
ax2.set_xlabel('X 좌표 (공격 방향 →)')
ax2.set_ylabel('Y 좌표')
ax2.axvline(x=88.5, color='green', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('reports/figures/reports/figures/shot_goal_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 슈팅/골 시각화 저장: shot_goal_analysis.png")

# 시간대별 득점 분포
print("\n[3.4.1] 시간대별 득점 분포")
time_goal_counts = goal_data['time_period'].value_counts().reindex(time_order)
print(time_goal_counts)



# 3.5 포지션별 분석
print("\n[3.5] 포지션별 이벤트 분석")
position_events = df.groupby('main_position')['type_name'].count().sort_values(ascending=False)
print(position_events)


# 3.6 핵심 선수 분석 (Player Ranking)
print("\n[3.6] 핵심 선수 분석 (Top 5)")

# 선수별 합계 통계 계산
player_stats = df.groupby(['player_name_ko', 'team_name_ko']).apply(
    lambda x: pd.Series({
        'pass_count': len(x[x['type_name'].isin(['Pass', 'Pass_Freekick', 'Cross'])]),
        'shot_count': len(x[x['type_name'].isin(['Shot', 'Shot_Freekick'])]),
        'goal_count': len(x[x['result_name'] == 'Goal'])
    })
).reset_index()

# 1. 패스왕 Top 5
print("\n[패스 횟수 Top 5]")
print(player_stats.sort_values('pass_count', ascending=False).head(5)[['player_name_ko', 'team_name_ko', 'pass_count']])

# 2. 슈팅왕 Top 5
print("\n[슈팅 횟수 Top 5]")
print(player_stats.sort_values('shot_count', ascending=False).head(5)[['player_name_ko', 'team_name_ko', 'shot_count']])

# 3. 득점왕 Top 5
print("\n[득점 Top 5]")
print(player_stats.sort_values('goal_count', ascending=False).head(5)[['player_name_ko', 'team_name_ko', 'goal_count']])

# =============================================================================
# 4. 통계 분석 (Statistical Analysis)
# =============================================================================
print("\n[4] 통계 분석")
print("-" * 40)

# 4.1 홈/어웨이 득점 차이 분석 (t-test)
print("\n[4.1] 홈/어웨이 득점 차이 분석")
home_scores = match_info['home_score']
away_scores = match_info['away_score']

t_stat, p_value = stats.ttest_rel(home_scores, away_scores)
print(f"홈팀 평균 득점: {home_scores.mean():.2f}")
print(f"어웨이팀 평균 득점: {away_scores.mean():.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("→ 홈/어웨이 득점 간 통계적으로 유의한 차이 있음 (p < 0.05)")
else:
    print("→ 홈/어웨이 득점 간 통계적으로 유의한 차이 없음 (p >= 0.05)")

# 4.2 패스 성공률과 승률 간 상관관계
print("\n[4.2] 상관관계 분석: 패스 성공률 vs 승률")

# 팀별 승률 계산
team_results = df.groupby(['team_name_ko', 'game_id']).first()['match_result'].reset_index()
team_win_rate = team_results.groupby('team_name_ko')['match_result'].apply(
    lambda x: (x == '승리').sum() / len(x) * 100
).to_frame('승률(%)')

# 패스 성공률과 병합
team_stats = team_pass_stats[['성공률(%)']].merge(team_win_rate, left_index=True, right_index=True)

# 상관계수 계산
corr, corr_p = stats.pearsonr(team_stats['성공률(%)'], team_stats['승률(%)'])
print(f"상관계수 (Pearson r): {corr:.4f}")
print(f"p-value: {corr_p:.4f}")
if corr_p < 0.05:
    print(f"→ 패스 성공률과 승률 간 통계적으로 유의한 상관관계 있음")
else:
    print(f"→ 패스 성공률과 승률 간 통계적으로 유의한 상관관계 없음")

# 상관관계 시각화
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(team_stats['성공률(%)'], team_stats['승률(%)'], s=100, alpha=0.7)
for idx, row in team_stats.iterrows():
    ax.annotate(idx, (row['성공률(%)'], row['승률(%)']), fontsize=9, ha='center', va='bottom')
    
# 추세선
z = np.polyfit(team_stats['성공률(%)'], team_stats['승률(%)'], 1)
p = np.poly1d(z)
ax.plot(team_stats['성공률(%)'].sort_values(), 
        p(team_stats['성공률(%)'].sort_values()), 
        "r--", alpha=0.7, label=f'추세선 (r={corr:.2f})')

ax.set_xlabel('패스 성공률 (%)', fontsize=12)
ax.set_ylabel('승률 (%)', fontsize=12)
ax.set_title('팀별 패스 성공률 vs 승률 상관관계', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/reports/figures/correlation_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 상관관계 시각화 저장: correlation_analysis.png")



# 4.2.1 경기 통계 간 상관관계 분석 (Correlation Matrix)
print("\n[4.2.1] 경기 통계 간 종합 상관관계 분석")

# [상관분석용] 경기별-팀별 요약 데이터 생성
pass_types_corr = ["Pass", "Pass_Freekick", "Cross", "Pass_Corner", "Throw-In"]
shot_types_corr = ["Shot", "Shot_Freekick", "Penalty", "Penalty Kick"]

def _game_team_agg(x: pd.DataFrame) -> pd.Series:
    pass_mask = x["type_name"].isin(pass_types_corr)
    total_passes = int(pass_mask.sum())
    successful_passes = int((pass_mask & x["result_name"].eq("Successful")).sum())
    forward_passes = int((pass_mask & (x["dx"] > 5)).sum())

    total_shots = int(x["type_name"].isin(shot_types_corr).sum())
    goals = int((x["type_name"] == "Goal").sum())

    return pd.Series(
        {
            "total_passes": total_passes,
            "pass_success_rate": (successful_passes / max(total_passes, 1)) * 100,
            "forward_pass_ratio": (forward_passes / max(total_passes, 1)) * 100,
            "total_shots": total_shots,
            "goals": goals,
            "tackles": int((x["type_name"] == "Tackle").sum()),
            "interceptions": int((x["type_name"] == "Interception").sum()),
            "fouls": int((x["type_name"] == "Foul").sum()),
            "attack_zone_actions": int((x["start_x"] > 70).sum()),  # 공격 진영 활동
            "take_ons": int((x["type_name"] == "Take On").sum()),  # 드리블 돌파
        }
    )

game_team_stats = df.groupby(["game_id", "team_name_ko"]).apply(_game_team_agg).reset_index()


# 경기별 팀 통계에서 수치형 데이터만 선택
corr_features = [
    "total_passes",
    "pass_success_rate",
    "forward_pass_ratio",
    "total_shots",
    "goals",
    "tackles",
    "interceptions",
    "fouls",
    "attack_zone_actions",
    "take_ons",
]

# 상관관계 행렬 계산
correlation_matrix = game_team_stats[corr_features].corr()
print("\n[상관관계 행렬]")
print(correlation_matrix.round(2))

# 히트맵 시각화
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, 
            annot=True,           # 숫자 표시
            fmt='.2f',            # 소수점 2자리
            cmap='RdBu_r',        # 빨강(음수) - 파랑(양수) 색상
            center=0,             # 0을 기준으로 색 분리
            square=True,          # 정사각형 셀
            linewidths=0.5,       # 셀 테두리
            ax=ax)

ax.set_title('경기 통계 간 상관관계 히트맵', fontsize=16)
plt.tight_layout()
plt.savefig('reports/figures/reports/figures/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 상관관계 히트맵 저장: correlation_heatmap.png")

# 주요 상관관계 인사이트 추출 (0.5 이상 또는 -0.5 이하)
print("\n[주요 상관관계 (|r| >= 0.5)]")
for i in range(len(corr_features)):
    for j in range(i+1, len(corr_features)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) >= 0.5:
            print(f"  • {corr_features[i]} ↔ {corr_features[j]}: r = {corr_val:.2f}")




# 4.3 포지션별 패스 성공률 차이 (ANOVA)
print("\n[4.3] 포지션별 패스 성공률 차이 분석 (ANOVA)")
position_pass = pass_data[pass_data['main_position'].notna()].copy()
position_pass['is_successful_num'] = position_pass['is_successful'].fillna(0)

# 포지션별 그룹화
positions = position_pass['main_position'].unique()
position_groups = [position_pass[position_pass['main_position'] == pos]['is_successful_num'].values 
                   for pos in positions if len(position_pass[position_pass['main_position'] == pos]) > 0]

# scipy.stats.f_oneway는 환경/바이너리 조합에 따라 비정상 종료(SIGFPE 등)가 보고될 수 있어
# 1-way ANOVA를 안전하게(넘파이 기반) 직접 계산합니다.
if len(position_groups) < 2:
    print("ANOVA 수행 불가: 비교할 포지션 그룹이 2개 미만입니다.")
    f_stat, anova_p = np.nan, np.nan
else:
    groups = [np.asarray(g, dtype=float) for g in position_groups if len(g) > 0]
    k = len(groups)
    n_total = int(sum(len(g) for g in groups))

    all_vals = np.concatenate(groups) if n_total > 0 else np.array([], dtype=float)
    grand_mean = float(all_vals.mean()) if n_total > 0 else np.nan

    ss_between = float(sum(len(g) * (float(g.mean()) - grand_mean) ** 2 for g in groups))
    ss_within = float(sum(((g - float(g.mean())) ** 2).sum() for g in groups))

    df_between = k - 1
    df_within = n_total - k
    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_within = ss_within / df_within if df_within > 0 else np.nan

    f_stat = ms_between / ms_within if (ms_within is not None and ms_within > 0) else np.nan
    anova_p = stats.f.sf(f_stat, df_between, df_within) if np.isfinite(f_stat) else np.nan

    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {anova_p:.6f}")
    if np.isfinite(anova_p) and anova_p < 0.05:
        print("→ 포지션별 패스 성공률에 통계적으로 유의한 차이 있음")
    else:
        print("→ 포지션별 패스 성공률에 통계적으로 유의한 차이 없음")

# 포지션별 패스 성공률 시각화
position_pass_rate = pass_data.groupby('main_position').apply(
    lambda x: (x['result_name'] == 'Successful').sum() / len(x) * 100
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
position_pass_rate.plot(kind='bar', ax=ax, color='teal')
ax.set_title('포지션별 패스 성공률', fontsize=14)
ax.set_xlabel('포지션')
ax.set_ylabel('패스 성공률 (%)')
ax.axhline(y=position_pass_rate.mean(), color='red', linestyle='--', label=f'평균: {position_pass_rate.mean():.1f}%')
ax.legend()
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('reports/figures/reports/figures/position_pass_rate.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 포지션별 패스 성공률 시각화 저장: position_pass_rate.png")

# 4.4 카이제곱 검정: 홈/어웨이 승률
print("\n[4.4] 홈/어웨이 승리 분포 (카이제곱 검정)")
home_wins = (match_info['home_score'] > match_info['away_score']).sum()
away_wins = (match_info['home_score'] < match_info['away_score']).sum()
draws = (match_info['home_score'] == match_info['away_score']).sum()

print(f"홈 승리: {home_wins}경기 ({home_wins/len(match_info)*100:.1f}%)")
print(f"어웨이 승리: {away_wins}경기 ({away_wins/len(match_info)*100:.1f}%)")
print(f"무승부: {draws}경기 ({draws/len(match_info)*100:.1f}%)")

observed = [home_wins, away_wins, draws]
expected = [len(match_info)/3] * 3  # 균등 분포 가정
chi2, chi_p = stats.chisquare(observed, expected)
print(f"\n카이제곱 통계량: {chi2:.4f}")
print(f"p-value: {chi_p:.4f}")
if chi_p < 0.05:
    print("→ 경기 결과 분포가 균등하지 않음 (홈 어드밴티지 존재 가능)")
else:
    print("→ 경기 결과 분포가 통계적으로 균등함")

# =============================================================================
# 5. 머신러닝 (Machine Learning)
# =============================================================================
print("\n[5] 머신러닝 분석")
print("-" * 40)

# =============================================================================
# 5. 머신러닝 (Machine Learning) - 업그레이드 버전
# =============================================================================
print("\n[5] 머신러닝 분석 (Advanced)")
print("-" * 40)

# 5.1 경기별 팀 통계 집계 (고급 지표 추가)
print("\n[5.1] 경기별 심화 통계 특성 생성")

def calculate_game_stats_advanced(game_df):
    """경기별 팀 통계 계산 (파생 변수 아이디어 반영)"""
    stats_dict = {}
    
    # 1. 기본 카운트
    # 패스 관련
    pass_df = game_df[game_df['type_name'].isin(['Pass', 'Pass_Freekick', 'Pass_Corner', 'Cross'])]
    total_passes = len(pass_df)
    stats_dict['total_passes'] = total_passes
    stats_dict['pass_success_rate'] = (pass_df['result_name'] == 'Successful').mean() * 100 if total_passes > 0 else 0
    
    # 슈팅 관련
    shot_df = game_df[game_df['type_name'].isin(['Shot', 'Shot_Freekick', 'Penalty Kick'])]
    total_shots = len(shot_df)
    stats_dict['total_shots'] = total_shots
    
    # 수비 관련
    stats_dict['defensive_actions'] = len(game_df[game_df['type_name'].isin(['Tackle', 'Interception', 'Clearance', 'Block'])])
    stats_dict['fouls'] = (game_df['type_name'] == 'Foul').sum()
    
    # 2. [NEW] 고급 파생 지표 반영
    # (1) 공격성 지표: 전진 패스 비율
    forward_passes = (game_df['pass_direction'] == '전진 패스').sum()
    stats_dict['forward_pass_ratio'] = forward_passes / total_passes * 100 if total_passes > 0 else 0
    
    # (2) 주도권 지표: 공격 지역(Attack Zone) 패스 비중
    attack_zone_passes = len(pass_df[pass_df['field_zone'] == '공격 1/3'])
    stats_dict['attack_pass_ratio'] = attack_zone_passes / total_passes * 100 if total_passes > 0 else 0
    
    # (3) 슈팅 집중도: 슈팅까지 연결된 효율 (슈팅 수 / 전체 패스 수)
    stats_dict['shot_creation_efficiency'] = total_shots / total_passes * 100 if total_passes > 0 else 0

    return pd.Series(stats_dict)

# 경기-팀별 통계 계산
print("경기별 심화 데이터 집계 중... (시간이 조금 걸릴 수 있습니다)")
game_team_stats = df.groupby(['game_id', 'team_id', 'team_name_ko']).apply(calculate_game_stats_advanced).reset_index()

# 경기 결과 및 홈/어웨이 정보 추가
game_results = df.groupby(['game_id', 'team_id']).first()[['match_result', 'home_team_id']].reset_index()
game_team_stats = game_team_stats.merge(game_results, on=['game_id', 'team_id'])

# [NEW] 홈 이점(Home Advantage) 변수 추가 (핵심!)
# 내 팀 ID와 홈 팀 ID가 같으면 1 (홈), 다르면 0 (어웨이)
game_team_stats['is_home'] = (game_team_stats['team_id'] == game_team_stats['home_team_id']).astype(int)

print(f"생성된 경기-팀 통계 수: {len(game_team_stats)}")

# 5.2 승/무/패 예측 모델
print("\n[5.2] 경기 결과 예측 모델 (Random Forest + 고급 지표)")

# 업그레이드된 특성 목록
# (goals는 데이터 누수 방지를 위해 제외했습니다. 승패의 '원인'만 넣어야 합니다.)
features = [
    'is_home',                 # 홈/어웨이 여부 (승률에 큰 영향)
    'pass_success_rate',       # 기초 전력
    'forward_pass_ratio',      # 공격성
    'attack_pass_ratio',       # [NEW] 공격 지역 장악력
    'shot_creation_efficiency',# [NEW] 공격 작업 효율
    'defensive_actions',       # [NEW] 수비 강도
    'fouls',                   # 거친 플레이 여부
    'take_ons'                 # 드리블 돌파 (기존 데이터 없으면 0 처리됨)
]

# 해당 컬럼이 있는지 확인 후 없는 건 0으로 채움
for col in features:
    if col not in game_team_stats.columns:
        game_team_stats[col] = 0

X = game_team_stats[features].copy()
y = game_team_stats['match_result'].copy()

# 결측치 처리
X = X.fillna(0)

# 라벨 인코딩
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest 모델 (옵션 튜닝)
rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

print("\n[Random Forest 분류 결과 (성능 향상 버전)]")
print(f"정확도: {accuracy_score(y_test, rf_pred)*100:.1f}%")
print("\n분류 리포트:")
print(classification_report(y_test, rf_pred, target_names=le.classes_))

# 특성 중요도 시각화
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
feature_importance.plot(kind='barh', x='feature', y='importance', ax=ax, color='forestgreen', legend=False)
ax.set_title('경기 결과 예측 - 특성 중요도 (Advanced)', fontsize=14)
ax.set_xlabel('중요도')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('reports/figures/reports/figures/feature_importance_adv.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 특성 중요도 시각화 저장: feature_importance_adv.png")



# 5.3 슈팅 성공(골) 예측 모델
print("\n[5.3] 슈팅 성공(골) 예측 모델 (Logistic Regression)")

# 슈팅 데이터 준비
shot_df = df[df['type_name'].isin(['Shot', 'Shot_Freekick', 'Penalty Kick', 'Goal'])].copy()

# 골 여부 (Goal이면 1, 아니면 0)
shot_df['is_goal'] = (shot_df['type_name'] == 'Goal').astype(int)

# 슈팅 거리 (골대까지)
shot_df['shot_distance'] = np.sqrt((105 - shot_df['start_x'])**2 + (34 - shot_df['start_y'])**2)

# 슈팅 각도 (간단 버전)
shot_df['shot_angle'] = np.abs(np.arctan2(shot_df['start_y'] - 34, 105 - shot_df['start_x'])) * 180 / np.pi

# 특성 선택
shot_features = ['start_x', 'start_y', 'shot_distance', 'shot_angle']
X_shot = shot_df[shot_features].dropna()
y_shot = shot_df.loc[X_shot.index, 'is_goal']

# 학습/테스트 분할
X_shot_train, X_shot_test, y_shot_train, y_shot_test = train_test_split(
    X_shot, y_shot, test_size=0.2, random_state=42, stratify=y_shot
)

# 스케일링
shot_scaler = StandardScaler()
X_shot_train_scaled = shot_scaler.fit_transform(X_shot_train)
X_shot_test_scaled = shot_scaler.transform(X_shot_test)

# Logistic Regression
lr_model = LogisticRegression(random_state=42, solver="liblinear", max_iter=200)
lr_model.fit(X_shot_train_scaled, y_shot_train)
lr_pred = lr_model.predict(X_shot_test_scaled)

print("\n[Logistic Regression 분류 결과]")
print(f"정확도: {accuracy_score(y_shot_test, lr_pred)*100:.1f}%")
print("\n분류 리포트:")
print(classification_report(y_shot_test, lr_pred, target_names=['No Goal', 'Goal']))

# 계수 분석
shot_coef = pd.DataFrame({
    'feature': shot_features,
    'coefficient': lr_model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print("\n[슈팅 성공 예측 - 계수]")
print(shot_coef)

# =============================================================================
# 6. 인사이트 추출 (Insights)
# =============================================================================
print("\n[6] 인사이트 추출")
print("-" * 40)

# 6.1 상위/하위 팀 분석
print("\n[6.1] 상위/하위 팀 특성 비교")

# 팀별 총 득점으로 순위
team_goals = match_info.copy()
team_goals_home = team_goals.groupby('home_team_name_ko')['home_score'].sum()
team_goals_away = team_goals.groupby('away_team_name_ko')['away_score'].sum()
total_goals = (team_goals_home.add(team_goals_away, fill_value=0)).sort_values(ascending=False)

print("\n[팀별 총 득점 순위]")
print(total_goals)

top_teams = total_goals.head(3).index.tolist()
bottom_teams = total_goals.tail(3).index.tolist()

print(f"\n상위 3팀: {top_teams}")
print(f"하위 3팀: {bottom_teams}")

# 상위/하위 팀 통계 비교
top_stats = game_team_stats[game_team_stats['team_name_ko'].isin(top_teams)][features].mean()
bottom_stats = game_team_stats[game_team_stats['team_name_ko'].isin(bottom_teams)][features].mean()

comparison = pd.DataFrame({
    '상위 3팀 평균': top_stats,
    '하위 3팀 평균': bottom_stats,
    '차이': top_stats - bottom_stats
})
print("\n[상위/하위 팀 통계 비교]")
print(comparison)

# 비교 시각화
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(features))
width = 0.35

bars1 = ax.bar(x - width/2, top_stats.values, width, label='상위 3팀', color='royalblue')
bars2 = ax.bar(x + width/2, bottom_stats.values, width, label='하위 3팀', color='indianred')

ax.set_xlabel('통계 지표')
ax.set_ylabel('평균값')
ax.set_title('상위 vs 하위 팀 경기 통계 비교', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('reports/figures/reports/figures/team_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 팀 비교 시각화 저장: team_comparison.png")

# 6.2 핵심 인사이트 요약
print("\n" + "=" * 60)
print("📊 K리그 2024 데이터 분석 핵심 인사이트")
print("=" * 60)

insights = []

# 인사이트 1: 홈 어드밴티지
home_win_pct = home_wins / len(match_info) * 100
if home_win_pct > 40:
    insights.append(f"🏠 홈 어드밴티지 존재: 홈팀 승률 {home_win_pct:.1f}% (K리그에서 홈 경기 중요)")

# 인사이트 2: 패스 성공률과 승률
if corr > 0.5:
    insights.append(f"⚽ 패스 성공률과 승률 강한 양의 상관관계 (r={corr:.2f}): 정확한 패싱이 승리의 핵심")
elif corr > 0.3:
    insights.append(f"⚽ 패스 성공률과 승률 중간 정도 상관관계 (r={corr:.2f})")

# 인사이트 3: 가장 중요한 예측 특성
top_feature = feature_importance.iloc[0]['feature']
insights.append(f"🎯 경기 결과 예측에 가장 중요한 지표: {top_feature}")

# 인사이트 4: 상위 팀 특징
top_advantage = comparison[comparison['차이'] > 0]['차이'].idxmax()
insights.append(f"🏆 상위 팀이 하위 팀 대비 가장 우수한 지표: {top_advantage}")

# 인사이트 5: 슈팅 위치
avg_goal_x = goal_positions['start_x'].mean() if len(goal_positions) > 0 else 0
insights.append(f"🥅 평균 골 위치 X좌표: {avg_goal_x:.1f} (페널티 박스 안에서 골 결정력 중요)")

# 인사이트 6: 포지션별 패스
best_pass_position = position_pass_rate.idxmax()
insights.append(f"📍 패스 성공률 최고 포지션: {best_pass_position} ({position_pass_rate.max():.1f}%)")

print("\n[핵심 인사이트]")
for i, insight in enumerate(insights, 1):
    print(f"{i}. {insight}")

# 6.3 추가 분석 제언
print("\n[추가 분석 제언]")
recommendations = [
    "1. 선수 개인별 심층 분석 (MVP 후보 선정용)",
    "2. 시계열 분석으로 팀 폼 변화 추적",
    "3. 상대 전적을 고려한 매치업 분석",
    "4. 세트피스(코너킥, 프리킥) 득점 효율 분석",
    "5. xG(Expected Goals) 모델 구축"
]
for rec in recommendations:
    print(f"  {rec}")

# =============================================================================
# 결과 요약 저장
# =============================================================================
print("\n" + "=" * 60)
print("분석 완료!")
print("=" * 60)
print("\n[생성된 파일]")
print("  - eda_overview.png: 기본 EDA 시각화")
print("  - shot_goal_analysis.png: 슈팅/골 분석")
print("  - correlation_analysis.png: 상관관계 분석")
print("  - position_pass_rate.png: 포지션별 패스 성공률")
print("  - feature_importance.png: 특성 중요도")
print("  - team_comparison.png: 상위/하위 팀 비교")

# 최종 요약 데이터프레임 저장
summary_data = {
    '총 경기 수': [match_info.shape[0]],
    '총 이벤트 수': [len(df)],
    '총 팀 수': [df['team_name_ko'].nunique()],
    '총 선수 수': [df['player_name_ko'].nunique()],
    '전체 패스 성공률(%)': [len(pass_success)/len(pass_data)*100],
    '홈팀 승률(%)': [home_win_pct],
    'ML 모델 정확도(%)': [accuracy_score(y_test, rf_pred)*100]
}
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('data/processed/analysis_summary.csv', index=False, encoding='utf-8-sig')
print("  - data/processed/analysis_summary.csv: 분석 요약 통계")

print("\n분석이 성공적으로 완료되었습니다! 🎉")
