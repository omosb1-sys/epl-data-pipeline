"""
📊 K리그 2024 시즌 고급 분석 - Part 2
=============================================
기존 노트북의 EDA 이후 분석 코드 (통계분석 → 머신러닝 → 인사이트 추출)

🎯 분석 구성:
  [1] 데이터 로드 및 파생컬럼 생성 (복습)
  [2] 고급 통계분석 (가설검정, ANOVA)
  [3] 상관관계 분석 (Correlation Matrix, Heatmap)
  [4] 머신러닝 모델링 (로지스틱 회귀, 랜덤포레스트)
  [5] 인사이트 추출 (팀 성과, 스타일 분류)
  [6] 시각화 및 보고서 생성

작성자: Claude (Senior Data Analyst)
난이도: ⭐⭐⭐ (초보자-중급자)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, accuracy_score, 
                            roc_auc_score, roc_curve, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 🔧 기본 설정
# ============================================================
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("\n" + "="*80)
print("🎯 K리그 2024 시즌 고급 분석 - Part 2 시작!")
print("   (통계분석, 상관관계, 머신러닝, 인사이트 추출)")
print("="*80 + "\n")


# ============================================================
# 📂 [1] 데이터 로드 및 파생컬럼 생성
# ============================================================
print("[1단계] 데이터 로드 및 파생컬럼 생성")
print("-"*80)

try:
    raw_data = pd.read_csv('data/raw/raw_data.csv', encoding='utf-8')
    match_info = pd.read_csv('data/raw/match_info.csv', encoding='utf-8')
    print(f"✓ raw_data 로드 성공: {raw_data.shape}")
    print(f"✓ match_info 로드 성공: {match_info.shape}")
except FileNotFoundError:
    print("❌ 데이터 파일을 찾을 수 없습니다.")
    exit(1)

# 데이터 타입 변환
match_info['game_date'] = pd.to_datetime(match_info['game_date'])

# 결측치 처리
raw_data['result_name'] = raw_data['result_name'].fillna('Unknown')

# 데이터 병합
df = raw_data.merge(match_info, on='game_id', how='left')
print(f"✓ 데이터 병합 완료: {df.shape}")

# 경기별-팀별 통계 데이터 생성 (Feature Engineering)
print("\n[1-2] 경기별-팀별 통계 데이터 생성")

game_team_stats = df.groupby(['game_id', 'team_name_ko']).apply(
    lambda x: pd.Series({
        # 기본 패스 지표
        'total_passes': len(x[x['type_name'].isin(['Pass', 'Pass_Freekick', 'Cross'])]),
        'pass_success_rate': (x['result_name'] == 'Successful').sum() / 
                             max(len(x[x['type_name'].isin(['Pass', 'Pass_Freekick', 'Cross'])]), 1) * 100,
        # 슈팅 지표
        'total_shots': len(x[x['type_name'].isin(['Shot', 'Shot_Freekick', 'Penalty'])]),
        'goals': len(x[x['type_name'] == 'Goal']),
        # 수비 지표
        'tackles': len(x[x['type_name'] == 'Tackle']),
        'interceptions': len(x[x['type_name'] == 'Interception']),
        'fouls': len(x[x['type_name'] == 'Foul']),
        # 공격 지표
        'attack_zone_actions': len(x[x['start_x'] > 70]),
        'take_ons': x['type_name'].astype(str).str.lower().str.contains('take').sum(),
        # 추가 지표
        'crosses': len(x[x['type_name'] == 'Cross']),
        'corners': len(x[x['type_name'] == 'Pass_Corner']),
    })
).reset_index()

# 경기 정보 merge (효율적인 방법)
game_team_stats = game_team_stats.merge(
    match_info[['game_id', 'home_score', 'away_score', 'home_team_id', 'away_team_id', 'game_date']], 
    on='game_id',
    how='left'
)

# team_id 추가
team_id_map = df.groupby(['game_id', 'team_name_ko'])['team_id'].first().reset_index()
game_team_stats = game_team_stats.merge(team_id_map, on=['game_id', 'team_name_ko'], how='left')

# 홈/어웨이 구분
game_team_stats['is_home'] = (game_team_stats['team_id'] == game_team_stats['home_team_id']).astype(int)

# 승리/무승부 여부 계산
def calculate_match_result(row):
    """경기 결과를 계산하는 함수"""
    if row['team_id'] == row['home_team_id']:
        our_score = row['home_score']
        their_score = row['away_score']
    else:
        our_score = row['away_score']
        their_score = row['home_score']
    
    if our_score > their_score:
        return 1  # 승리
    elif our_score == their_score:
        return 1  # 무승부 (점수 획득)
    else:
        return 0  # 패배

game_team_stats['win_or_draw'] = game_team_stats.apply(calculate_match_result, axis=1)

# 상대팀 실점 정보
def get_goals_against(row):
    """상대팀 득점(=내 실점) 계산"""
    if row['team_id'] == row['home_team_id']:
        return row['away_score']
    else:
        return row['home_score']

game_team_stats['goals_against'] = game_team_stats.apply(get_goals_against, axis=1)

# 득실차 계산
game_team_stats['goal_diff'] = game_team_stats['goals'] - game_team_stats['goals_against']

print(f"✓ 통계 데이터 생성 완료: {game_team_stats.shape}")
print(f"✓ 생성된 컬럼: {list(game_team_stats.columns)}")


# ============================================================
# 📊 [2] 고급 통계분석 (Statistical Analysis)
# ============================================================
print("\n" + "="*80)
print("[2단계] 고급 통계분석")
print("="*80)

"""
💡 통계분석의 목적:
   "데이터에서 발견한 차이가 정말 의미있는 차이인가?"
   를 수학적으로 증명하는 것입니다.

📌 p-value 해석:
   - p-value < 0.05: 통계적으로 유의미함 ✓ (95% 확신)
   - p-value ≥ 0.05: 유의미하지 않음 ✗
"""

# [2-1] 홈팀 vs 어웨이팀 득점 차이 (독립표본 t-검정)
print("\n[2-1] 가설검정: 홈팀 vs 어웨이팀 득점 차이")
print("-"*60)

home_goals = game_team_stats[game_team_stats['is_home'] == 1]['goals']
away_goals = game_team_stats[game_team_stats['is_home'] == 0]['goals']

print(f"홈팀 평균 득점: {home_goals.mean():.2f}골 (±{home_goals.std():.2f})")
print(f"어웨이팀 평균 득점: {away_goals.mean():.2f}골 (±{away_goals.std():.2f})")

# 정규성 검정 (Shapiro-Wilk)
_, p_shapiro_home = stats.shapiro(home_goals[:50])  # 샘플 제한
_, p_shapiro_away = stats.shapiro(away_goals[:50])

print(f"\n📋 정규성 검정 (Shapiro-Wilk):")
print(f"  홈팀: p={p_shapiro_home:.4f} {'✓ 정규분포' if p_shapiro_home > 0.05 else '✗ 비정규분포'}")
print(f"  어웨이팀: p={p_shapiro_away:.4f} {'✓ 정규분포' if p_shapiro_away > 0.05 else '✗ 비정규분포'}")

# 등분산 검정 (Levene)
_, p_levene = stats.levene(home_goals, away_goals)
print(f"\n📋 등분산 검정 (Levene): p={p_levene:.4f} {'✓ 분산 동일' if p_levene > 0.05 else '✗ 분산 다름'}")

# 독립표본 t-검정
t_stat, p_ttest = stats.ttest_ind(home_goals, away_goals)
print(f"\n⭐ 독립표본 t-검정 결과:")
print(f"  t-통계량: {t_stat:.4f}")
print(f"  p-value: {p_ttest:.4f}")

if p_ttest < 0.05:
    home_advantage = (home_goals.mean() - away_goals.mean()) / away_goals.mean() * 100
    print(f"  ✓ 결론: 홈팀과 어웨이팀의 득점에 통계적으로 유의미한 차이가 있습니다!")
    print(f"  📈 홈 이점: {home_advantage:.1f}%")
else:
    print(f"  ✗ 결론: 홈팀과 어웨이팀의 득점에 유의미한 차이가 없습니다.")


# [2-2] 일원분산분석 (ANOVA) - 팀별 평균 득점 비교
print("\n[2-2] 일원분산분석 (ANOVA): 팀별 평균 득점 비교")
print("-"*60)

"""
💡 ANOVA란?
   여러 그룹(팀)의 평균을 비교할 때 사용합니다.
   "모든 팀의 평균 득점이 같은가?" 를 검정합니다.
"""

team_goals_list = [group['goals'].values for name, group in game_team_stats.groupby('team_name_ko')]
f_stat, p_anova = stats.f_oneway(*team_goals_list)

print(f"F-통계량: {f_stat:.4f}")
print(f"p-value: {p_anova:.4f}")

if p_anova < 0.05:
    print(f"✓ 결론: 팀별로 평균 득점에 통계적으로 유의미한 차이가 있습니다!")
else:
    print(f"✗ 결론: 팀별 평균 득점에 유의미한 차이가 없습니다.")


# [2-3] 팀별 평균 득점 순위
print("\n🏆 팀별 평균 득점 순위:")
team_goal_ranking = game_team_stats.groupby('team_name_ko').agg({
    'goals': ['mean', 'std', 'count'],
    'goals_against': 'mean'
}).round(2)
team_goal_ranking.columns = ['평균득점', '표준편차', '경기수', '평균실점']
team_goal_ranking = team_goal_ranking.sort_values('평균득점', ascending=False)

for idx, (team, row) in enumerate(team_goal_ranking.iterrows(), 1):
    print(f"  {idx:2d}. {team}: {row['평균득점']:.2f}골 (±{row['표준편차']:.2f})")


# [2-4] 패스 성공률과 승률 간 상관관계 검정
print("\n[2-4] 상관관계 검정: 패스 성공률 vs 승률")
print("-"*60)

# 팀별 패스 성공률 및 승률 계산
team_performance = game_team_stats.groupby('team_name_ko').agg({
    'pass_success_rate': 'mean',
    'win_or_draw': 'mean'
}).round(2)
team_performance.columns = ['패스성공률', '승률']
team_performance['승률'] = team_performance['승률'] * 100

# 피어슨 상관계수
corr_pass_win, p_corr = stats.pearsonr(team_performance['패스성공률'], team_performance['승률'])
print(f"상관계수 (Pearson r): {corr_pass_win:.4f}")
print(f"p-value: {p_corr:.4f}")

if p_corr < 0.05:
    direction = "양의" if corr_pass_win > 0 else "음의"
    print(f"✓ 결론: 패스 성공률과 승률 간 통계적으로 유의미한 {direction} 상관관계가 있습니다!")
else:
    print(f"✗ 결론: 패스 성공률과 승률 간 유의미한 상관관계가 없습니다.")


# ============================================================
# 📈 [3] 상관관계 분석 (Correlation Analysis)
# ============================================================
print("\n" + "="*80)
print("[3단계] 상관관계 분석")
print("="*80)

# [3-1] 경기 통계 간 상관관계 매트릭스
print("\n[3-1] 경기 통계 간 상관관계 매트릭스")
print("-"*60)

corr_features = ['total_passes', 'pass_success_rate', 'total_shots', 
                 'goals', 'tackles', 'interceptions', 'fouls', 
                 'attack_zone_actions', 'take_ons']

# 수치형으로 변환 및 결측치 처리
for col in corr_features:
    game_team_stats[col] = pd.to_numeric(game_team_stats[col], errors='coerce').fillna(0)

# 상관관계 행렬 계산
correlation_matrix = game_team_stats[corr_features].corr()

print("\n[Pearson 상관관계 행렬]")
print(correlation_matrix.round(2))

# [3-2] 득점(goals)과 다른 변수들 간의 상관관계
print("\n[3-2] 득점(goals)과의 상관관계 분석")
print("-"*60)

goal_correlations = correlation_matrix['goals'].sort_values(ascending=False)
print("\n득점과의 상관관계 순위:")
for var, corr in goal_correlations.items():
    if var != 'goals':
        strength = "강함" if abs(corr) > 0.5 else ("중간" if abs(corr) > 0.3 else "약함")
        direction = "↑" if corr > 0 else "↓"
        print(f"  {direction} {var:25s}: {corr:6.3f} ({strength})")

# [3-3] 상관관계 히트맵 시각화
print("\n[3-3] 상관관계 히트맵 생성 중...")

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, 
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('경기 통계 간 상관관계 분석', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/reports/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 저장됨: correlation_heatmap.png")


# ============================================================
# 🤖 [4] 머신러닝 모델링 (Machine Learning)
# ============================================================
print("\n" + "="*80)
print("[4단계] 머신러닝: 승리 예측 모델")
print("="*80)

"""
💡 머신러닝이란?
   데이터의 패턴을 학습하여 미래를 예측하는 기법입니다.
   
   예) 과거 경기의 패스 성공률, 슈팅 수, 태클 수를 보고
      → 이번 경기 승리 여부를 예측

📌 분류 문제 (Classification):
   - 승리/무승부 (1) vs 패배 (0) 를 예측
   - 이진 분류 문제입니다
"""

# [4-1] 데이터 준비
print("\n[4-1] 데이터 준비")
print("-"*60)

feature_cols = [
    'total_passes', 'pass_success_rate', 'total_shots',
    'tackles', 'interceptions', 'fouls', 
    'attack_zone_actions', 'take_ons', 'is_home'
]

X = game_team_stats[feature_cols].fillna(0)
y = game_team_stats['win_or_draw']

print(f"피처 수: {X.shape[1]}")
print(f"샘플 수: {X.shape[0]}")
print(f"승리/무승부 비율: {y.mean()*100:.1f}%")
print(f"사용 피처: {', '.join(feature_cols)}")

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n학습 데이터: {X_train.shape[0]}개, 테스트 데이터: {X_test.shape[0]}개")

# 피처 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ 피처 정규화 완료")


# [4-2] 모델 1: 로지스틱 회귀
print("\n[4-2] 모델 1: 로지스틱 회귀 (Logistic Regression)")
print("-"*60)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

accuracy_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)

print(f"✓ 모델 학습 완료!")
print(f"\n📊 성능 지표:")
print(f"  정확도 (Accuracy): {accuracy_lr:.4f} ({accuracy_lr*100:.1f}%)")
print(f"  AUC 점수: {auc_lr:.4f}")

print(f"\n📋 분류 성능 보고서:")
print(classification_report(y_test, y_pred_lr, 
                          target_names=['패배', '승리/무승부'],
                          zero_division=0))

# 피처 중요도 (계수)
feature_importance_lr = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': lr_model.coef_[0]
})
feature_importance_lr['Abs_Coefficient'] = np.abs(feature_importance_lr['Coefficient'])
feature_importance_lr = feature_importance_lr.sort_values('Abs_Coefficient', ascending=False)

print("\n📈 피처 중요도 (로지스틱 회귀):")
for idx, row in feature_importance_lr.iterrows():
    direction = "➕" if row['Coefficient'] > 0 else "➖"
    print(f"  {direction} {row['Feature']:25s}: {row['Coefficient']:7.4f}")


# [4-3] 모델 2: 랜덤포레스트
print("\n[4-3] 모델 2: 랜덤포레스트 (Random Forest)")
print("-"*60)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                  random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

accuracy_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print(f"✓ 모델 학습 완료!")
print(f"\n📊 성능 지표:")
print(f"  정확도 (Accuracy): {accuracy_rf:.4f} ({accuracy_rf*100:.1f}%)")
print(f"  AUC 점수: {auc_rf:.4f}")

print(f"\n📋 분류 성능 보고서:")
print(classification_report(y_test, y_pred_rf, 
                          target_names=['패배', '승리/무승부'],
                          zero_division=0))

# 피처 중요도
feature_importance_rf = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📈 피처 중요도 (랜덤포레스트):")
for idx, row in feature_importance_rf.iterrows():
    bar = "█" * int(row['Importance'] * 50)
    print(f"  {row['Feature']:25s}: {bar} {row['Importance']:.4f}")


# [4-4] 모델 비교
print("\n[4-4] 모델 성능 비교")
print("-"*60)

model_comparison = pd.DataFrame({
    '모델': ['로지스틱 회귀', '랜덤포레스트'],
    '정확도': [accuracy_lr, accuracy_rf],
    'AUC': [auc_lr, auc_rf]
})
print(model_comparison.to_string(index=False))

best_model_name = model_comparison.loc[model_comparison['AUC'].idxmax(), '모델']
best_auc = model_comparison['AUC'].max()
print(f"\n⭐ 최고 성능 모델: {best_model_name} (AUC: {best_auc:.4f})")


# [4-5] 교차 검증
print("\n[4-5] 교차 검증 (5-Fold Cross Validation)")
print("-"*60)

cv_scores_lr = cross_val_score(
    LogisticRegression(max_iter=1000, random_state=42),
    X_train_scaled, y_train, cv=5, scoring='accuracy'
)

cv_scores_rf = cross_val_score(
    RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    X_train, y_train, cv=5, scoring='accuracy'
)

print(f"로지스틱 회귀:")
print(f"  평균 정확도: {cv_scores_lr.mean():.4f} (±{cv_scores_lr.std():.4f})")

print(f"\n랜덤포레스트:")
print(f"  평균 정확도: {cv_scores_rf.mean():.4f} (±{cv_scores_rf.std():.4f})")


# ============================================================
# 💡 [5] 인사이트 추출 (Insight Extraction)
# ============================================================
print("\n" + "="*80)
print("[5단계] 인사이트 추출 및 분석")
print("="*80)

# [5-1] 팀 순위 및 성과
print("\n[5-1] 팀 성과 분석 및 최종 순위")
print("-"*60)

team_stats_full = game_team_stats.groupby('team_name_ko').agg({
    'win_or_draw': ['sum', 'count'],
    'goals': ['sum', 'mean'],
    'goals_against': ['sum', 'mean'],
    'total_passes': 'mean',
    'pass_success_rate': 'mean',
    'total_shots': 'mean'
}).round(2)

team_stats_full.columns = ['승/무', '경기수', '총득점', '평균득점', 
                           '총실점', '평균실점', '평균패스', '패스성공률', '평균슈팅']

# 승점 계산 (승/무=1, 패=0 이므로 승/무 수 * 3 + 패 수 * 0 로 단순화)
# 여기서는 win_or_draw가 승리와 무승부를 합친 것이므로 근사치
team_stats_full['득실차'] = team_stats_full['총득점'] - team_stats_full['총실점']
team_stats_full = team_stats_full.sort_values(['승/무', '득실차'], ascending=[False, False])

print("\n🏆 2024 K리그 팀 성과 순위")
print(team_stats_full[['경기수', '승/무', '총득점', '총실점', '득실차', '평균득점', '평균실점']].to_string())


# [5-2] 팀별 플레이 스타일 분류
print("\n[5-2] 팀별 플레이 스타일 분류")
print("-"*60)

"""
💡 팀 스타일 분류 기준:
   - 공격형: 슈팅 많고 공격지역 액션 많음
   - 수비형: 태클/인터셉션 많음
   - 점유형: 패스 많고 성공률 높음
   - 올라운더: 모든 지표가 평균 이상
"""

team_style = game_team_stats.groupby('team_name_ko').agg({
    'total_shots': 'mean',
    'goals': 'mean',
    'tackles': 'mean',
    'interceptions': 'mean',
    'total_passes': 'mean',
    'pass_success_rate': 'mean',
    'attack_zone_actions': 'mean'
}).round(2)

# 지표 정규화 (평균 대비)
team_style['attack_index'] = (team_style['total_shots'] / team_style['total_shots'].mean() + 
                              team_style['goals'] / team_style['goals'].mean()) / 2
team_style['defense_index'] = (team_style['tackles'] / team_style['tackles'].mean() + 
                               team_style['interceptions'] / team_style['interceptions'].mean()) / 2
team_style['possession_index'] = (team_style['total_passes'] / team_style['total_passes'].mean() + 
                                   team_style['pass_success_rate'] / team_style['pass_success_rate'].mean()) / 2

def classify_style(row):
    """팀 스타일 분류"""
    if row['attack_index'] > 1.1 and row['defense_index'] > 1.1:
        return '🌟 올라운더'
    elif row['attack_index'] > 1.1:
        return '⚽ 공격형'
    elif row['defense_index'] > 1.1:
        return '🛡️ 수비형'
    elif row['possession_index'] > 1.1:
        return '🎯 점유형'
    else:
        return '⚖️ 균형형'

team_style['style'] = team_style.apply(classify_style, axis=1)

print("\n팀별 플레이 스타일:")
for style in ['🌟 올라운더', '⚽ 공격형', '🛡️ 수비형', '🎯 점유형', '⚖️ 균형형']:
    teams = team_style[team_style['style'] == style].index.tolist()
    if teams:
        print(f"\n{style}:")
        for team in teams:
            row = team_style.loc[team]
            print(f"  • {team}: 공격={row['attack_index']:.2f}, 수비={row['defense_index']:.2f}, 점유={row['possession_index']:.2f}")


# [5-3] 홈 어드밴티지 분석
print("\n[5-3] 홈 필드 어드밴티지 분석")
print("-"*60)

home_stats = game_team_stats[game_team_stats['is_home'] == 1]
away_stats = game_team_stats[game_team_stats['is_home'] == 0]

home_win_rate = home_stats['win_or_draw'].mean() * 100
away_win_rate = away_stats['win_or_draw'].mean() * 100

print(f"홈팀 승리/무승부율: {home_win_rate:.1f}%")
print(f"어웨이팀 승리/무승부율: {away_win_rate:.1f}%")
print(f"홈 어드밴티지: {home_win_rate - away_win_rate:.1f}%p")

print(f"\n📊 홈 vs 어웨이 상세 비교:")
comparison = pd.DataFrame({
    '홈팀': [home_stats['goals'].mean(), home_stats['total_shots'].mean(), 
             home_stats['pass_success_rate'].mean(), home_stats['tackles'].mean()],
    '어웨이팀': [away_stats['goals'].mean(), away_stats['total_shots'].mean(),
               away_stats['pass_success_rate'].mean(), away_stats['tackles'].mean()]
}, index=['평균득점', '평균슈팅', '패스성공률', '평균태클'])
print(comparison.round(2).to_string())


# [5-4] 핵심 인사이트 요약
print("\n[5-4] 핵심 인사이트 요약")
print("-"*60)

# 가장 중요한 피처 추출
top_feature = feature_importance_rf.iloc[0]['Feature']
top_importance = feature_importance_rf.iloc[0]['Importance']

# 가장 득점 효율 좋은 팀
best_scoring_team = team_goal_ranking.index[0]
best_scoring_avg = team_goal_ranking.iloc[0]['평균득점']

# 가장 수비 좋은 팀
best_defense_team = team_goal_ranking.sort_values('평균실점').index[0]
best_defense_avg = team_goal_ranking.sort_values('평균실점').iloc[0]['평균실점']

print(f"""
🔍 핵심 인사이트:

1️⃣ 승리 예측에 가장 중요한 요소: {top_feature} (중요도: {top_importance:.3f})

2️⃣ 최다 득점팀: {best_scoring_team} (평균 {best_scoring_avg:.2f}골)

3️⃣ 최소 실점팀: {best_defense_team} (평균 {best_defense_avg:.2f}골)

4️⃣ 홈 어드밴티지: {home_win_rate - away_win_rate:.1f}%p (홈경기 시 유리)

5️⃣ 머신러닝 예측 정확도: {max(accuracy_lr, accuracy_rf)*100:.1f}%
""")


# ============================================================
# 📈 [6] 시각화 및 보고서 생성
# ============================================================
print("\n" + "="*80)
print("[6단계] 시각화 및 보고서 생성")
print("="*80)

# [6-1] ROC 곡선
print("\n[6-1] ROC 곡선 생성 중...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 로지스틱 회귀 ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
axes[0].plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={auc_lr:.3f})', linewidth=2, color='blue')
axes[0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC 곡선 - 로지스틱 회귀')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 랜덤포레스트 ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
axes[1].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_rf:.3f})', linewidth=2, color='orange')
axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC 곡선 - 랜덤포레스트')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/reports/figures/ml_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 저장됨: ml_roc_curves.png")


# [6-2] 피처 중요도 비교
print("[6-2] 피처 중요도 비교 시각화 중...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 로지스틱 회귀
axes[0].barh(feature_importance_lr['Feature'], feature_importance_lr['Coefficient'], color='steelblue')
axes[0].set_xlabel('계수값')
axes[0].set_title('피처 중요도 - 로지스틱 회귀', fontsize=14)
axes[0].grid(True, alpha=0.3, axis='x')

# 랜덤포레스트
axes[1].barh(feature_importance_rf['Feature'], feature_importance_rf['Importance'], color='coral')
axes[1].set_xlabel('중요도')
axes[1].set_title('피처 중요도 - 랜덤포레스트', fontsize=14)
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('reports/figures/reports/figures/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 저장됨: feature_importance_comparison.png")


# [6-3] 팀 성과 종합 시각화
print("[6-3] 팀 성과 종합 시각화 중...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 팀별 득점/실점
ax1 = axes[0, 0]
x_pos = np.arange(len(team_stats_full))
width = 0.35
ax1.bar(x_pos - width/2, team_stats_full['평균득점'], width, label='평균득점', color='steelblue')
ax1.bar(x_pos + width/2, team_stats_full['평균실점'], width, label='평균실점', color='coral')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(team_stats_full.index, rotation=45, ha='right')
ax1.set_ylabel('골')
ax1.set_title('팀별 평균 득점/실점', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 팀 스타일 산점도
ax2 = axes[0, 1]
scatter = ax2.scatter(team_style['attack_index'], team_style['defense_index'], 
                      s=150, alpha=0.7, c=team_style['possession_index'], cmap='RdYlGn')
for team in team_style.index:
    ax2.annotate(team, (team_style.loc[team, 'attack_index'], 
                        team_style.loc[team, 'defense_index']),
                fontsize=9, ha='center', va='bottom')
ax2.axhline(y=1, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=1, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('공격 지수')
ax2.set_ylabel('수비 지수')
ax2.set_title('팀 스타일 분류 (공격/수비)', fontsize=14)
plt.colorbar(scatter, ax=ax2, label='점유 지수')

# 홈 vs 어웨이 비교
ax3 = axes[1, 0]
home_away_data = pd.DataFrame({
    '홈팀': [home_stats['goals'].mean(), home_stats['total_shots'].mean(), 
             home_stats['pass_success_rate'].mean()],
    '어웨이팀': [away_stats['goals'].mean(), away_stats['total_shots'].mean(),
               away_stats['pass_success_rate'].mean()]
}, index=['평균득점', '평균슈팅', '패스성공률'])
home_away_data.plot(kind='bar', ax=ax3, color=['skyblue', 'salmon'])
ax3.set_ylabel('값')
ax3.set_title('홈 vs 어웨이 성능 비교', fontsize=14)
ax3.tick_params(axis='x', rotation=0)
ax3.grid(True, alpha=0.3, axis='y')
ax3.legend(title='위치')

# 혼동 행렬 (랜덤포레스트)
ax4 = axes[1, 1]
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['패배', '승리/무승부'], yticklabels=['패배', '승리/무승부'])
ax4.set_xlabel('예측')
ax4.set_ylabel('실제')
ax4.set_title('혼동 행렬 (랜덤포레스트)', fontsize=14)

plt.tight_layout()
plt.savefig('reports/figures/reports/figures/team_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 저장됨: team_performance_analysis.png")


# [6-4] 분포 분석 시각화
print("[6-4] 분포 분석 시각화 중...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, col, color, title in zip(
    axes.flat,
    ['goals', 'total_shots', 'pass_success_rate', 'tackles'],
    ['steelblue', 'coral', 'green', 'purple'],
    ['득점 분포', '슈팅 수 분포', '패스 성공률 분포', '태클 수 분포']
):
    ax.hist(game_team_stats[col], bins=20, color=color, edgecolor='black', alpha=0.7)
    ax.axvline(game_team_stats[col].mean(), color='red', linestyle='--', linewidth=2, label='평균')
    ax.set_xlabel(col)
    ax.set_ylabel('빈도')
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/reports/figures/distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 저장됨: distribution_analysis.png")


# ============================================================
# 📄 [7] 최종 보고서 생성
# ============================================================
print("\n[7단계] 최종 보고서 생성")
print("-"*80)

report = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                   2024 K리그 고급 분석 최종 보고서                             ║
║                   Part 2: 통계분석 ~ 머신러닝 ~ 인사이트                       ║
╚════════════════════════════════════════════════════════════════════════════════╝


【 1. 분석 개요 】
──────────────────────────────────────────────────────────────────────────────
총 경기 수: {game_team_stats['game_id'].nunique()}경기
분석 팀: {game_team_stats['team_name_ko'].nunique()}개 팀
총 데이터 포인트: {len(game_team_stats)}개 (경기-팀별)


【 2. 통계 검정 결과 】
──────────────────────────────────────────────────────────────────────────────

2-1) 홈/어웨이 득점 차이 (t-검정)
    • 홈팀 평균 득점: {home_goals.mean():.2f}골
    • 어웨이팀 평균 득점: {away_goals.mean():.2f}골
    • t-통계량: {t_stat:.4f}
    • p-value: {p_ttest:.4f}
    • 결론: {'✓ 유의미한 차이 있음' if p_ttest < 0.05 else '✗ 유의미한 차이 없음'}

2-2) 팀별 득점 차이 (ANOVA)
    • F-통계량: {f_stat:.4f}
    • p-value: {p_anova:.4f}
    • 결론: {'✓ 팀별 유의미한 차이 있음' if p_anova < 0.05 else '✗ 팀별 차이 없음'}

2-3) 패스 성공률 ↔ 승률 상관관계
    • Pearson r: {corr_pass_win:.4f}
    • p-value: {p_corr:.4f}
    • 결론: {'✓ 유의미한 상관관계' if p_corr < 0.05 else '✗ 상관관계 없음'}


【 3. 머신러닝 모델 성과 】
──────────────────────────────────────────────────────────────────────────────

3-1) 모델 성능 비교
┌──────────────────┬──────────────┬────────────┐
│ 모델             │ 정확도(%)    │ AUC 점수   │
├──────────────────┼──────────────┼────────────┤
│ 로지스틱 회귀    │ {accuracy_lr*100:6.2f}%     │ {auc_lr:.4f}     │
│ 랜덤포레스트     │ {accuracy_rf*100:6.2f}%     │ {auc_rf:.4f}     │
└──────────────────┴──────────────┴────────────┘

3-2) 최고 성능 모델: {best_model_name}
    • AUC: {best_auc:.4f}
    • 교차 검증 정확도: {(cv_scores_lr.mean() if auc_lr > auc_rf else cv_scores_rf.mean()):.4f}

3-3) 핵심 피처 (Top 5)
"""

for i, (idx, row) in enumerate(feature_importance_rf.head(5).iterrows(), 1):
    report += f"    {i}. {row['Feature']}: {row['Importance']:.4f}\n"

report += f"""

【 4. 팀별 스타일 분류 】
──────────────────────────────────────────────────────────────────────────────
"""

for style in ['🌟 올라운더', '⚽ 공격형', '🛡️ 수비형', '🎯 점유형', '⚖️ 균형형']:
    teams = team_style[team_style['style'] == style].index.tolist()
    if teams:
        report += f"\n{style}: {', '.join(teams)}"

report += f"""


【 5. 핵심 인사이트 】
──────────────────────────────────────────────────────────────────────────────

✓ 완성된 분석:
  • 가설검정 (t-검정, ANOVA)
  • 상관관계 분석 (Pearson)
  • 머신러닝 예측 모델 (정확도 {max(accuracy_lr, accuracy_rf)*100:.1f}%)
  • 팀별 스타일 분류
  • 홈/어웨이 이점 분석

💡 데이터 기반 인사이트:
  1. 승리 예측에 가장 중요한 요소: {top_feature}
  2. 최다 득점팀: {best_scoring_team} (평균 {best_scoring_avg:.2f}골)
  3. 최소 실점팀: {best_defense_team} (평균 {best_defense_avg:.2f}골)
  4. 홈 어드밴티지: {home_win_rate - away_win_rate:.1f}%p


【 6. 생성된 파일 목록 】
──────────────────────────────────────────────────────────────────────────────
  📈 시각화:
     • correlation_heatmap.png - 상관관계 히트맵
     • ml_roc_curves.png - ROC 곡선 비교
     • feature_importance_comparison.png - 피처 중요도 비교
     • team_performance_analysis.png - 팀 성과 종합
     • distribution_analysis.png - 데이터 분포
  📄 보고서:
     • reports/docs/final_analysis_report.txt - 이 보고서
  📊 데이터:
     • team_ranking.csv - 팀 순위
     • feature_importance.csv - 피처 중요도

════════════════════════════════════════════════════════════════════════════════
                           분석 완료 | 2024 K리그 시즌
════════════════════════════════════════════════════════════════════════════════
"""

print(report)

# 보고서 저장
with open('reports/docs/final_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("✓ 분석 보고서 저장: reports/docs/final_analysis_report.txt")

# 팀 순위 저장
team_stats_full.to_csv('team_ranking.csv', encoding='utf-8')
print("✓ 팀 순위 저장: team_ranking.csv")

# 피처 중요도 저장
feature_importance_rf.to_csv('feature_importance.csv', index=False, encoding='utf-8')
print("✓ 피처 중요도 저장: feature_importance.csv")


# ============================================================
# 🎉 최종 결과
# ============================================================
print("\n" + "="*80)
print("✓✓✓ 모든 분석이 완료되었습니다! ✓✓✓")
print("="*80)

print(f"""
📊 생성된 결과물 요약:
  
  📈 시각화 (5개):
     • correlation_heatmap.png
     • ml_roc_curves.png
     • feature_importance_comparison.png
     • team_performance_analysis.png
     • distribution_analysis.png
     
  📄 보고서: reports/docs/final_analysis_report.txt
  
  📊 데이터: team_ranking.csv, feature_importance.csv

🎓 학습 포인트:
  1. 통계분석: t-검정, ANOVA, 상관관계 분석
  2. 머신러닝: 로지스틱 회귀, 랜덤포레스트
  3. 모델 평가: 정확도, AUC, 교차 검증
  4. 인사이트 추출: 팀 스타일, 홈 어드밴티지
  5. 시각화: matplotlib, seaborn 활용

🚀 실행 방법:
   python k_league_advanced_analysis.py
""")

print("="*80)
print("분석 완료! 수고하셨습니다! 🎉")
print("="*80 + "\n")
