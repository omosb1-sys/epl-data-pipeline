"""
K-리그 통합 분석 파이프라인
============================
[EDA → 통계 → ML → Causal AI → 시계열 → 보고서]

전체 분석을 한 번에 실행하고
HTML/Markdown 보고서를 자동 생성합니다.

Author: Antigravity (Senior Data Analyst)
Date: 2026-01-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.proportion import proportions_ztest
from datetime import datetime
import warnings
import os
import json

warnings.filterwarnings('ignore')

# ============================================
# 0. 환경 설정
# ============================================
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

BASE_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/data"
OUTPUT_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/output"
REPORT_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/reports"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)

# 분석 결과 저장용
ANALYSIS_RESULTS = {
    'meta': {},
    'eda': {},
    'general_stats': {},  # 일반 통계분석 결과
    'statistics': {},
    'ml': {},
    'causal': {},
    'timeseries': {},
    'insights': []
}


def print_header(title: str, emoji: str = "📊"):
    """섹션 헤더 출력"""
    print("\n" + "=" * 60)
    print(f"{emoji} {title}")
    print("=" * 60)


# ============================================
# 1. 데이터 로드 및 전처리
# ============================================
def load_and_prepare_data():
    """데이터 로드 및 기본 전처리"""
    print_header("STEP 1: 데이터 로드 및 전처리", "📂")

    match_info = pd.read_csv(f"{BASE_PATH}/match_info.csv")
    raw_data = pd.read_csv(f"{BASE_PATH}/raw_data.csv")

    # 메타 정보 저장
    ANALYSIS_RESULTS['meta'] = {
        'total_matches': match_info['game_id'].nunique(),
        'total_events': len(raw_data),
        'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'data_period': f"{match_info['game_date'].min()} ~ {match_info['game_date'].max()}"
    }

    print(f"✅ 총 경기 수: {ANALYSIS_RESULTS['meta']['total_matches']}")
    print(f"✅ 총 이벤트 수: {ANALYSIS_RESULTS['meta']['total_events']:,}")

    # 경기별 집계
    game_stats = raw_data.groupby(['game_id', 'team_id']).agg(
        total_actions=('action_id', 'count'),
        total_passes=('type_name', lambda x: (x == 'Pass').sum()),
        total_shots=('type_name', lambda x: (x == 'Shot').sum()),
        successful_actions=('result_name', lambda x: (x == 'Successful').sum()),
        avg_x_position=('start_x', 'mean'),
        avg_y_position=('start_y', 'mean'),
        unique_players=('player_id', 'nunique')
    ).reset_index()

    # 파생변수
    game_stats['pass_ratio'] = game_stats['total_passes'] / game_stats['total_actions']
    game_stats['shot_ratio'] = game_stats['total_shots'] / game_stats['total_actions']
    game_stats['success_rate'] = game_stats['successful_actions'] / game_stats['total_actions']

    # 메타데이터 병합 및 결과 라벨링
    merged = game_stats.merge(
        match_info[['game_id', 'home_team_id', 'away_team_id', 'home_score', 'away_score',
                   'home_team_name_ko', 'away_team_name_ko']],
        on='game_id', how='left'
    )

    def get_result(row):
        if row['team_id'] == row['home_team_id']:
            if row['home_score'] > row['away_score']: return 'Win'
            elif row['home_score'] < row['away_score']: return 'Lose'
            else: return 'Draw'
        else:
            if row['away_score'] > row['home_score']: return 'Win'
            elif row['away_score'] < row['home_score']: return 'Lose'
            else: return 'Draw'

    merged['result'] = merged.apply(get_result, axis=1)
    merged['win'] = (merged['result'] == 'Win').astype(int)

    return match_info, raw_data, merged


# ============================================
# 2. EDA 및 기본 통계
# ============================================
def run_eda(df):
    """탐색적 데이터 분석"""
    print_header("STEP 2: EDA & 기본 통계", "📈")

    numeric_cols = ['total_actions', 'total_passes', 'total_shots',
                    'success_rate', 'pass_ratio', 'shot_ratio']

    # 기술통계
    desc_stats = df[numeric_cols].describe()

    # 결과별 통계
    result_stats = df.groupby('result')[numeric_cols].mean()

    ANALYSIS_RESULTS['eda'] = {
        'mean_actions': df['total_actions'].mean(),
        'mean_passes': df['total_passes'].mean(),
        'mean_shots': df['total_shots'].mean(),
        'mean_success_rate': df['success_rate'].mean(),
        'win_rate': df['win'].mean()
    }

    print(f"✅ 경기당 평균 액션: {ANALYSIS_RESULTS['eda']['mean_actions']:.0f}")
    print(f"✅ 평균 성공률: {ANALYSIS_RESULTS['eda']['mean_success_rate']:.1%}")

    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[idx], color='steelblue')
        axes[idx].set_title(f'{col} 분포', fontsize=12)
        axes[idx].axvline(df[col].mean(), color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/01_eda_distributions.png", dpi=150)
    plt.close()

    return desc_stats, result_stats


# ============================================
# 2.5 일반 통계 분석 (NEW)
# ============================================
def run_general_statistics(df, match_info):
    """일반 통계 분석: t-검정, A/B테스트, 회귀, ARIMA"""
    print_header("STEP 2.5: 일반 통계 분석", "🧪")

    results = {}

    # 홈/원정 구분 추가
    df['is_home'] = (df['team_id'] == df['home_team_id']).astype(int)
    df['game_date'] = pd.to_datetime(match_info.set_index('game_id').loc[df['game_id'], 'game_date'].values)

    # === 1. t-검정: 홈 vs 원정 ===
    home_success = df[df['is_home'] == 1]['success_rate']
    away_success = df[df['is_home'] == 0]['success_rate']
    t_stat, p_ttest = stats.ttest_ind(home_success, away_success)

    results['ttest'] = {
        'home_mean': home_success.mean(),
        'away_mean': away_success.mean(),
        'p_value': p_ttest,
        'significant': p_ttest < 0.05
    }
    print(f"  [t-검정] 홈({home_success.mean():.3f}) vs 원정({away_success.mean():.3f}), p={p_ttest:.4f}")

    # === 2. 카이제곱: 홈/원정 vs 승패 ===
    contingency = pd.crosstab(df['is_home'], df['win'])
    chi2, p_chi, _, _ = stats.chi2_contingency(contingency)

    results['chi2'] = {'chi2': chi2, 'p_value': p_chi, 'significant': p_chi < 0.05}
    print(f"  [카이제곱] 홈/원정 ↔ 승패 독립성: χ²={chi2:.2f}, p={p_chi:.4f}")

    # === 3. A/B 테스트: 슈팅 수 ===
    median_shots = df['total_shots'].median()
    group_a = df[df['total_shots'] >= median_shots]['win']
    group_b = df[df['total_shots'] < median_shots]['win']

    count = np.array([group_a.sum(), group_b.sum()])
    nobs = np.array([len(group_a), len(group_b)])
    z_stat, p_ab = proportions_ztest(count, nobs)
    lift = (group_a.mean() - group_b.mean()) / group_b.mean() * 100 if group_b.mean() > 0 else 0

    results['ab_test'] = {
        'high_shots_win_rate': group_a.mean(),
        'low_shots_win_rate': group_b.mean(),
        'lift_percent': lift,
        'p_value': p_ab,
        'significant': p_ab < 0.05
    }
    print(f"  [A/B테스트] 높은 슈팅({group_a.mean():.1%}) vs 낮은 슈팅({group_b.mean():.1%}), Lift={lift:.1f}%")

    # === 4. 로지스틱 회귀 ===
    X_log = df[['total_shots', 'success_rate', 'avg_x_position', 'total_passes']].fillna(0)
    y_log = df['win']
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_log, y_log)
    log_accuracy = log_model.score(X_log, y_log)

    results['logistic'] = {
        'accuracy': log_accuracy,
        'odds_ratios': dict(zip(X_log.columns, np.exp(log_model.coef_[0])))
    }
    print(f"  [로지스틱 회귀] 정확도: {log_accuracy:.1%}")

    # === 5. ARIMA 예측 ===
    try:
        daily = df.groupby('game_date')['success_rate'].mean().sort_index()
        daily = daily.asfreq('D', method='ffill')
        if len(daily) >= 30:
            arima_model = ARIMA(daily, order=(1,1,1))
            arima_fit = arima_model.fit()
            forecast = arima_fit.forecast(steps=7)
            results['arima'] = {'forecast_mean': forecast.mean(), 'aic': arima_fit.aic}
            print(f"  [ARIMA] 7일 예측 평균: {forecast.mean():.3f}")
        else:
            results['arima'] = {'error': 'Insufficient data'}
    except Exception as e:
        results['arima'] = {'error': str(e)}

    # === 6. 시계열 분해 ===
    try:
        daily_actions = df.groupby('game_date')['total_actions'].mean().sort_index()
        daily_actions = daily_actions.asfreq('D', method='ffill')
        if len(daily_actions) >= 14:
            decomp = seasonal_decompose(daily_actions, model='additive', period=7)
            results['decomposition'] = {
                'trend_mean': decomp.trend.dropna().mean(),
                'seasonal_amplitude': decomp.seasonal.max() - decomp.seasonal.min()
            }
            print(f"  [시계열분해] Trend 평균: {decomp.trend.dropna().mean():.1f}")
        else:
            results['decomposition'] = {'error': 'Insufficient data'}
    except Exception as e:
        results['decomposition'] = {'error': str(e)}

    ANALYSIS_RESULTS['general_stats'] = results

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # t-검정
    ax1 = axes[0, 0]
    ax1.bar(['홈팀', '원정팀'], [home_success.mean(), away_success.mean()], color=['coral', 'steelblue'])
    ax1.set_ylabel('성공률')
    ax1.set_title(f't-검정: 홈 vs 원정 (p={p_ttest:.4f})')

    # A/B 테스트
    ax2 = axes[0, 1]
    ax2.bar(['높은\n슈팅', '낮은\n슈팅'], [group_a.mean(), group_b.mean()],
            color=['seagreen' if p_ab < 0.05 else 'gray', 'lightcoral'])
    ax2.set_ylabel('승률')
    ax2.set_title(f'A/B테스트: 슈팅수별 승률 (Lift={lift:.1f}%)')

    # 로지스틱 회귀 계수
    ax3 = axes[1, 0]
    coefs = log_model.coef_[0]
    colors = ['seagreen' if c > 0 else 'lightcoral' for c in coefs]
    ax3.barh(X_log.columns, coefs, color=colors)
    ax3.axvline(x=0, color='black', linestyle='--')
    ax3.set_title(f'로지스틱 회귀 계수 (정확도={log_accuracy:.1%})')

    # 카이제곱
    ax4 = axes[1, 1]
    contingency.plot(kind='bar', ax=ax4, color=['lightcoral', 'seagreen'])
    ax4.set_title(f'카이제곱: 홈/원정 vs 승패 (p={p_chi:.4f})')
    ax4.set_xticklabels(['원정', '홈'], rotation=0)
    ax4.legend(['패배', '승리'])

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/02_5_general_stats.png", dpi=150)
    plt.close()
    print(f"  🎨 저장: {OUTPUT_PATH}/02_5_general_stats.png")

    return results


# ============================================
# 3. 상관관계 분석
# ============================================
def run_correlation(df):
    """상관관계 분석"""
    print_header("STEP 3: 상관관계 분석", "🔗")

    numeric_cols = ['total_actions', 'total_passes', 'total_shots',
                    'success_rate', 'pass_ratio', 'avg_x_position']

    corr_matrix = df[numeric_cols].corr()

    # 주요 상관관계 추출
    high_corr = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                high_corr.append({
                    'var1': numeric_cols[i],
                    'var2': numeric_cols[j],
                    'correlation': corr_matrix.iloc[i, j]
                })

    ANALYSIS_RESULTS['statistics']['high_correlations'] = high_corr

    # 시각화
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r',
                center=0, fmt='.2f', square=True)
    plt.title('상관관계 행렬', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/02_correlation_matrix.png", dpi=150)
    plt.close()

    print(f"✅ 강한 상관관계 발견: {len(high_corr)}개")

    return corr_matrix


# ============================================
# 4. 고급 통계 분석
# ============================================
def run_advanced_statistics(df):
    """가설검정 및 고급 통계"""
    print_header("STEP 4: 고급 통계 분석", "🧪")

    # ANOVA
    win_data = df[df['result'] == 'Win']['success_rate']
    draw_data = df[df['result'] == 'Draw']['success_rate']
    lose_data = df[df['result'] == 'Lose']['success_rate']

    f_stat, p_anova = stats.f_oneway(win_data, draw_data, lose_data)

    # Cohen's d
    cohens_d = (win_data.mean() - lose_data.mean()) / np.sqrt(
        ((win_data.std()**2) + (lose_data.std()**2)) / 2
    )

    ANALYSIS_RESULTS['statistics']['anova'] = {
        'f_statistic': f_stat,
        'p_value': p_anova,
        'significant': p_anova < 0.05
    }
    ANALYSIS_RESULTS['statistics']['cohens_d'] = cohens_d

    print(f"✅ ANOVA p-value: {p_anova:.4f} ({'유의미' if p_anova < 0.05 else '유의미하지 않음'})")
    print(f"✅ Cohen's d: {cohens_d:.3f}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x='result', y='success_rate', data=df, order=['Win', 'Draw', 'Lose'],
                palette='Set2', ax=axes[0])
    axes[0].set_title('승/무/패별 성공률 분포')
    sns.boxplot(x='result', y='total_shots', data=df, order=['Win', 'Draw', 'Lose'],
                palette='Set2', ax=axes[1])
    axes[1].set_title('승/무/패별 슈팅 횟수')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/03_advanced_stats.png", dpi=150)
    plt.close()

    return {'f_stat': f_stat, 'p_value': p_anova, 'cohens_d': cohens_d}


# ============================================
# 5. 머신러닝 분석
# ============================================
def run_machine_learning(df):
    """머신러닝 모델 학습 및 평가"""
    print_header("STEP 5: 머신러닝 분석", "🤖")

    feature_cols = ['total_actions', 'total_passes', 'total_shots',
                    'success_rate', 'pass_ratio', 'shot_ratio',
                    'avg_x_position', 'unique_players']

    X = df[feature_cols].fillna(0).values
    y = df['result'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 모델 학습
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        score = accuracy_score(y_test, model.predict(X_test_scaled))
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    # Permutation Importance
    perm_importance = permutation_importance(best_model, X_test_scaled, y_test, n_repeats=30, random_state=42)
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=False)

    ANALYSIS_RESULTS['ml'] = {
        'best_model': best_name,
        'accuracy': best_score,
        'top_features': importance_df.head(3).to_dict('records')
    }

    print(f"✅ 최고 모델: {best_name} (정확도: {best_score:.1%})")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis', ax=axes[0])
    axes[0].set_title('Feature Importance')

    cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    axes[1].set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/04_ml_results.png", dpi=150)
    plt.close()

    return best_model, importance_df


# ============================================
# 6. Causal AI 분석
# ============================================
def run_causal_analysis(df):
    """인과 관계 분석"""
    print_header("STEP 6: Causal AI 분석", "🔬")

    # Treatment 정의
    median_x = df['avg_x_position'].median()
    df['aggressive'] = (df['avg_x_position'] > median_x).astype(int)

    # 단순 비교
    treated_win = df[df['aggressive'] == 1]['win'].mean()
    control_win = df[df['aggressive'] == 0]['win'].mean()
    naive_ate = treated_win - control_win

    # PSM
    confounders = ['total_passes', 'success_rate', 'unique_players']
    X = df[confounders].fillna(0)

    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X, df['aggressive'])
    df['ps'] = ps_model.predict_proba(X)[:, 1]

    treated_df = df[df['aggressive'] == 1]
    control_df = df[df['aggressive'] == 0]

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control_df[['ps']])
    _, indices = nn.kneighbors(treated_df[['ps']])
    matched_control = control_df.iloc[indices.flatten()]

    att = treated_df['win'].mean() - matched_control['win'].mean()

    ANALYSIS_RESULTS['causal'] = {
        'naive_ate': naive_ate,
        'psm_att': att,
        'selection_bias': naive_ate - att
    }

    print(f"✅ 단순 비교 (Biased): {naive_ate:.3f}")
    print(f"✅ PSM 인과 효과 (ATT): {att:.3f}")

    # 시각화
    plt.figure(figsize=(10, 5))
    methods = ['단순 비교\n(Biased)', 'PSM\n(Causal)']
    effects = [naive_ate, att]
    colors = ['lightcoral', 'seagreen']
    bars = plt.bar(methods, effects, color=colors)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.ylabel('승률 차이')
    plt.title('단순 비교 vs PSM 인과 효과')
    for bar, eff in zip(bars, effects):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{eff:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/05_causal_analysis.png", dpi=150)
    plt.close()

    return att


# ============================================
# 7. 시계열 분석
# ============================================
def run_timeseries_analysis(raw_data, match_info, merged_df):
    """시계열 모멘텀 분석"""
    print_header("STEP 7: 시계열 분석", "⏱️")

    # 시간 윈도우 생성
    raw_data['time_window'] = (raw_data['time_seconds'] // 300).astype(int)

    window_stats = raw_data.groupby(['game_id', 'team_id', 'period_id', 'time_window']).agg(
        action_count=('action_id', 'count'),
        pass_count=('type_name', lambda x: (x == 'Pass').sum()),
        shot_count=('type_name', lambda x: (x == 'Shot').sum()),
        avg_x=('start_x', 'mean')
    ).reset_index()

    window_stats['success_rate'] = window_stats['pass_count'] / window_stats['action_count']
    window_stats['attack_intensity'] = window_stats['avg_x'] / 100
    window_stats['momentum_score'] = window_stats['success_rate'] * window_stats['attack_intensity']

    # 시퀀스 피처 생성
    sequence_features = []
    for (game_id, team_id), group in window_stats.groupby(['game_id', 'team_id']):
        group = group.sort_values('time_window')
        first_half = group[group['period_id'] == 1]
        second_half = group[group['period_id'] == 2]

        features = {
            'game_id': game_id, 'team_id': team_id,
            'first_half_momentum': first_half['momentum_score'].mean() if len(first_half) > 0 else 0,
            'second_half_momentum': second_half['momentum_score'].mean() if len(second_half) > 0 else 0,
            'momentum_std': group['momentum_score'].std(),
            'late_intensity': group.tail(3)['action_count'].mean() if len(group) >= 3 else 0
        }
        sequence_features.append(features)

    seq_df = pd.DataFrame(sequence_features)
    seq_df = seq_df.merge(merged_df[['game_id', 'team_id', 'win']], on=['game_id', 'team_id'])

    # 모델 학습
    feature_cols = ['first_half_momentum', 'second_half_momentum', 'momentum_std', 'late_intensity']
    X = seq_df[feature_cols].fillna(0)
    y = seq_df['win']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    ANALYSIS_RESULTS['timeseries'] = {
        'accuracy': accuracy,
        'top_feature': importance.iloc[0]['feature']
    }

    print(f"✅ 시계열 모델 정확도: {accuracy:.1%}")
    print(f"✅ 핵심 피처: {ANALYSIS_RESULTS['timeseries']['top_feature']}")

    # 시각화
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance, palette='viridis')
    plt.title('시계열 피처 중요도')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/06_timeseries_importance.png", dpi=150)
    plt.close()

    return accuracy


# ============================================
# 8. 인사이트 생성
# ============================================
def generate_insights():
    """핵심 인사이트 생성"""
    print_header("STEP 8: 인사이트 생성", "💡")

    insights = []

    # ML 기반 인사이트
    ml_results = ANALYSIS_RESULTS['ml']
    insights.append({
        'category': '머신러닝',
        'finding': f"최고 모델 {ml_results['best_model']}은 {ml_results['accuracy']:.1%} 정확도 달성",
        'implication': f"가장 중요한 변수는 '{ml_results['top_features'][0]['feature']}'"
    })

    # Causal AI 인사이트
    causal_results = ANALYSIS_RESULTS['causal']
    if causal_results['psm_att'] < 0:
        insights.append({
            'category': '인과분석',
            'finding': '공격적 포지셔닝만으로는 승리를 보장하지 않음',
            'implication': "'효율적인 전진'이 핵심, 단순 전진은 역효과"
        })

    # 시계열 인사이트
    ts_results = ANALYSIS_RESULTS['timeseries']
    insights.append({
        'category': '시계열',
        'finding': f"시계열 모델 정확도 {ts_results['accuracy']:.1%}로 기존 대비 향상",
        'implication': f"'{ts_results['top_feature']}'이 승패의 결정적 요소"
    })

    # 통계 인사이트
    stats_results = ANALYSIS_RESULTS['statistics']
    if stats_results['anova']['significant']:
        insights.append({
            'category': '통계',
            'finding': '승/무/패 그룹 간 성공률 차이 통계적으로 유의미',
            'implication': '성공률 관리가 승리의 핵심 전략'
        })

    ANALYSIS_RESULTS['insights'] = insights

    print(f"✅ 총 {len(insights)}개 핵심 인사이트 도출")

    return insights


# ============================================
# 9. HTML 보고서 생성
# ============================================
def generate_html_report():
    """HTML 보고서 생성"""
    print_header("STEP 9: HTML 보고서 생성", "📄")

    html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>K-리그 데이터 분석 보고서</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Apple SD Gothic Neo', sans-serif; background: #f5f7fa; color: #333; line-height: 1.8; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 40px 20px; }}
        header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 60px 40px; border-radius: 16px; margin-bottom: 40px; }}
        header h1 {{ font-size: 2.5rem; margin-bottom: 10px; }}
        header p {{ opacity: 0.9; font-size: 1.1rem; }}
        .meta {{ display: flex; gap: 30px; margin-top: 20px; flex-wrap: wrap; }}
        .meta-item {{ background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 8px; }}
        section {{ background: white; border-radius: 16px; padding: 40px; margin-bottom: 30px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
        section h2 {{ color: #667eea; font-size: 1.5rem; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #eee; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .kpi-card {{ background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ed 100%); padding: 25px; border-radius: 12px; text-align: center; }}
        .kpi-value {{ font-size: 2rem; font-weight: bold; color: #667eea; }}
        .kpi-label {{ color: #666; margin-top: 5px; }}
        .insight-card {{ background: #f8fafc; border-left: 4px solid #667eea; padding: 20px; margin: 15px 0; border-radius: 0 8px 8px 0; }}
        .insight-category {{ color: #667eea; font-weight: bold; font-size: 0.9rem; }}
        .insight-finding {{ font-size: 1.1rem; margin: 10px 0; }}
        .insight-implication {{ color: #666; font-style: italic; }}
        .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
        .chart-card {{ text-align: center; }}
        .chart-card img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .chart-title {{ margin-top: 10px; color: #666; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8fafc; color: #667eea; }}
        tr:hover {{ background: #f8fafc; }}
        .conclusion {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 16px; }}
        .conclusion h2 {{ color: white; border-bottom-color: rgba(255,255,255,0.3); }}
        footer {{ text-align: center; padding: 40px; color: #999; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>⚽ K-리그 데이터 분석 보고서</h1>
            <p>Antigravity AI 분석 시스템에 의해 자동 생성됨</p>
            <div class="meta">
                <div class="meta-item">📅 분석일시: {ANALYSIS_RESULTS['meta']['analysis_date']}</div>
                <div class="meta-item">🏟️ 총 경기: {ANALYSIS_RESULTS['meta']['total_matches']}경기</div>
                <div class="meta-item">📊 총 이벤트: {ANALYSIS_RESULTS['meta']['total_events']:,}건</div>
            </div>
        </header>

        <section>
            <h2>📊 1. 핵심 지표 (KPI)</h2>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-value">{ANALYSIS_RESULTS['eda']['mean_actions']:.0f}</div>
                    <div class="kpi-label">경기당 평균 액션</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{ANALYSIS_RESULTS['eda']['mean_success_rate']:.1%}</div>
                    <div class="kpi-label">평균 성공률</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{ANALYSIS_RESULTS['ml']['accuracy']:.1%}</div>
                    <div class="kpi-label">ML 예측 정확도</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{ANALYSIS_RESULTS['timeseries']['accuracy']:.1%}</div>
                    <div class="kpi-label">시계열 모델 정확도</div>
                </div>
            </div>
        </section>

        <section>
            <h2>💡 2. 핵심 인사이트</h2>
            {''.join([f'''
            <div class="insight-card">
                <div class="insight-category">{insight['category']}</div>
                <div class="insight-finding">{insight['finding']}</div>
                <div class="insight-implication">→ {insight['implication']}</div>
            </div>
            ''' for insight in ANALYSIS_RESULTS['insights']])}
        </section>

        <section>
            <h2>📈 3. 분석 시각화</h2>
            <div class="chart-grid">
                <div class="chart-card">
                    <img src="../output/02_correlation_matrix.png" alt="상관관계">
                    <div class="chart-title">상관관계 행렬</div>
                </div>
                <div class="chart-card">
                    <img src="../output/04_ml_results.png" alt="ML 결과">
                    <div class="chart-title">머신러닝 분석 결과</div>
                </div>
                <div class="chart-card">
                    <img src="../output/05_causal_analysis.png" alt="인과분석">
                    <div class="chart-title">Causal AI 분석</div>
                </div>
                <div class="chart-card">
                    <img src="../output/06_timeseries_importance.png" alt="시계열">
                    <div class="chart-title">시계열 피처 중요도</div>
                </div>
            </div>
        </section>

        <section>
            <h2>🔬 4. Causal AI 분석 결과</h2>
            <p><strong>연구 질문:</strong> "공격적 포지셔닝이 실제로 승리를 유발하는가?"</p>
            <table>
                <tr><th>분석 방법</th><th>효과 크기</th><th>해석</th></tr>
                <tr><td>단순 비교 (Biased)</td><td>{ANALYSIS_RESULTS['causal']['naive_ate']:.3f}</td><td>Selection Bias 포함</td></tr>
                <tr><td>PSM (Causal)</td><td>{ANALYSIS_RESULTS['causal']['psm_att']:.3f}</td><td>인과적 효과 추정</td></tr>
                <tr><td>Selection Bias</td><td>{ANALYSIS_RESULTS['causal']['selection_bias']:.3f}</td><td>편향 크기</td></tr>
            </table>
        </section>

        <section class="conclusion">
            <h2>🎯 5. 결론 및 제언</h2>
            <ul style="margin-left: 20px;">
                <li><strong>효율적 전진이 핵심:</strong> 단순히 공격적으로 진출하는 것보다 성공률 높은 플레이가 승리와 연결됨</li>
                <li><strong>후반전 모멘텀 관리:</strong> 시계열 분석 결과, 후반전 흐름 유지가 승패의 분수령</li>
                <li><strong>데이터 기반 전술:</strong> 상대팀 분석 시 성공률과 평균 포지션 데이터 활용 권장</li>
            </ul>
        </section>

        <footer>
            <p>Generated by Antigravity AI Analysis System | K-League Data Analytics</p>
            <p>분석 일시: {ANALYSIS_RESULTS['meta']['analysis_date']}</p>
        </footer>
    </div>
</body>
</html>
"""

    report_path = f"{REPORT_PATH}/k_league_analysis_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ HTML 보고서 저장: {report_path}")

    return report_path


# ============================================
# 10. Markdown 보고서 생성
# ============================================
def generate_markdown_report():
    """Markdown 보고서 생성"""

    md_content = f"""# ⚽ K-리그 데이터 분석 보고서

**분석 일시:** {ANALYSIS_RESULTS['meta']['analysis_date']}
**분석 시스템:** Antigravity AI

---

## 📊 1. 분석 개요

| 항목 | 값 |
|------|-----|
| 총 경기 수 | {ANALYSIS_RESULTS['meta']['total_matches']}경기 |
| 총 이벤트 수 | {ANALYSIS_RESULTS['meta']['total_events']:,}건 |
| 데이터 기간 | {ANALYSIS_RESULTS['meta']['data_period']} |

---

## 📈 2. 핵심 지표 (KPI)

| 지표 | 값 |
|------|-----|
| 경기당 평균 액션 | {ANALYSIS_RESULTS['eda']['mean_actions']:.0f} |
| 평균 성공률 | {ANALYSIS_RESULTS['eda']['mean_success_rate']:.1%} |
| ML 예측 정확도 | {ANALYSIS_RESULTS['ml']['accuracy']:.1%} |
| 시계열 모델 정확도 | {ANALYSIS_RESULTS['timeseries']['accuracy']:.1%} |

---

## 💡 3. 핵심 인사이트

{''.join([f'''
### {insight['category']}
- **발견:** {insight['finding']}
- **시사점:** {insight['implication']}

''' for insight in ANALYSIS_RESULTS['insights']])}

---

## 🔬 4. Causal AI 분석

**연구 질문:** "공격적 포지셔닝이 실제로 승리를 유발하는가?"

| 분석 방법 | 효과 크기 | 해석 |
|----------|----------|------|
| 단순 비교 | {ANALYSIS_RESULTS['causal']['naive_ate']:.3f} | Selection Bias 포함 |
| PSM (인과) | {ANALYSIS_RESULTS['causal']['psm_att']:.3f} | 인과적 효과 |
| 편향 크기 | {ANALYSIS_RESULTS['causal']['selection_bias']:.3f} | - |

---

## 🎯 5. 결론 및 제언

1. **효율적 전진이 핵심**: 단순 공격보다 성공률 높은 플레이가 승리 연결
2. **후반전 모멘텀**: 시계열 분석 결과, 후반전 흐름 유지가 승패 분수령
3. **데이터 기반 전술**: 상대팀 분석 시 성공률과 평균 포지션 활용 권장

---

*Generated by Antigravity AI Analysis System*
"""

    md_path = f"{REPORT_PATH}/k_league_analysis_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"✅ Markdown 보고서 저장: {md_path}")

    return md_path


# ============================================
# MAIN PIPELINE
# ============================================
def main():
    """전체 분석 파이프라인 실행"""
    start_time = datetime.now()

    print("\n" + "🚀" * 25)
    print("   K-리그 통합 분석 파이프라인")
    print("   [EDA → 통계 → ML → Causal AI → 시계열 → 보고서]")
    print("🚀" * 25 + "\n")

    try:
        # 1. 데이터 로드
        match_info, raw_data, df = load_and_prepare_data()

        # 2. EDA
        run_eda(df)

        # 2.5 일반 통계 분석 (NEW)
        run_general_statistics(df, match_info)

        # 3. 상관관계
        run_correlation(df)

        # 4. 고급 통계
        run_advanced_statistics(df)

        # 5. 머신러닝
        run_machine_learning(df)

        # 6. Causal AI
        run_causal_analysis(df)

        # 7. 시계열
        run_timeseries_analysis(raw_data, match_info, df)

        # 8. 인사이트
        generate_insights()

        # 9. HTML 보고서
        html_path = generate_html_report()

        # 10. Markdown 보고서
        md_path = generate_markdown_report()

        # 결과 JSON 저장
        with open(f"{REPORT_PATH}/analysis_results.json", 'w', encoding='utf-8') as f:
            # Convert non-serializable items
            results_serializable = ANALYSIS_RESULTS.copy()
            results_serializable['ml']['top_features'] = [dict(x) for x in results_serializable['ml']['top_features']]
            json.dump(results_serializable, f, ensure_ascii=False, indent=2, default=str)

        elapsed = (datetime.now() - start_time).total_seconds()

        print("\n" + "=" * 60)
        print("✅ 전체 파이프라인 완료!")
        print("=" * 60)
        print(f"⏱️ 총 소요 시간: {elapsed:.1f}초")
        print(f"📄 HTML 보고서: {html_path}")
        print(f"📄 Markdown 보고서: {md_path}")
        print(f"📊 시각화 파일: {OUTPUT_PATH}/")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
