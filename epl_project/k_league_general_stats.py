"""
K-리그 일반 통계분석 모듈
===========================
고급 분석 전 필수 기초 통계분석

1. t-검정 / 카이제곱 검정
2. A/B 테스트 (홈 vs 원정, 전반 vs 후반)
3. 회귀 분석 (선형/로지스틱)
4. ARIMA 시계열 예측
5. 시계열 분해 (Trend, Seasonality, Residual)

Author: Antigravity (Senior Data Analyst)
Date: 2026-01-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.proportion import proportions_ztest
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================
# 0. 환경 설정
# ============================================
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

BASE_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/data"
OUTPUT_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

STATS_RESULTS = {}


def print_header(title: str, emoji: str = "📊"):
    print("\n" + "=" * 60)
    print(f"{emoji} {title}")
    print("=" * 60)


def load_data():
    """데이터 로드"""
    print_header("데이터 로드", "📂")
    
    match_info = pd.read_csv(f"{BASE_PATH}/match_info.csv")
    raw_data = pd.read_csv(f"{BASE_PATH}/raw_data.csv")
    
    # 경기별 집계
    game_stats = raw_data.groupby(['game_id', 'team_id']).agg(
        total_actions=('action_id', 'count'),
        total_passes=('type_name', lambda x: (x == 'Pass').sum()),
        total_shots=('type_name', lambda x: (x == 'Shot').sum()),
        successful_actions=('result_name', lambda x: (x == 'Successful').sum()),
        avg_x_position=('start_x', 'mean')
    ).reset_index()
    
    game_stats['success_rate'] = game_stats['successful_actions'] / game_stats['total_actions']
    
    merged = game_stats.merge(
        match_info[['game_id', 'home_team_id', 'away_team_id', 'home_score', 'away_score', 'game_date']],
        on='game_id', how='left'
    )
    
    merged['is_home'] = (merged['team_id'] == merged['home_team_id']).astype(int)
    
    def get_result(row):
        if row['team_id'] == row['home_team_id']:
            return 1 if row['home_score'] > row['away_score'] else 0
        else:
            return 1 if row['away_score'] > row['home_score'] else 0
    
    merged['win'] = merged.apply(get_result, axis=1)
    merged['game_date'] = pd.to_datetime(merged['game_date'])
    
    print(f"✅ 데이터 Shape: {merged.shape}")
    
    return match_info, raw_data, merged


# ============================================
# 1. 기본 가설검정 (t-검정, 카이제곱)
# ============================================
def run_hypothesis_tests(df):
    """기본 가설검정"""
    print_header("STEP 1: 기본 가설검정 (t-검정, 카이제곱)", "🧪")
    
    results = {}
    
    # 1-1. 독립표본 t-검정: 홈팀 vs 원정팀 성공률
    home_success = df[df['is_home'] == 1]['success_rate']
    away_success = df[df['is_home'] == 0]['success_rate']
    
    t_stat, p_value = stats.ttest_ind(home_success, away_success)
    
    results['ttest_home_away'] = {
        'home_mean': home_success.mean(),
        'away_mean': away_success.mean(),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    print(f"\n[t-검정: 홈 vs 원정 성공률]")
    print(f"  • 홈팀 평균 성공률: {home_success.mean():.3f}")
    print(f"  • 원정팀 평균 성공률: {away_success.mean():.3f}")
    print(f"  • t-통계량: {t_stat:.3f}, p-value: {p_value:.4f}")
    print(f"  → {'유의미한 차이 있음' if p_value < 0.05 else '유의미한 차이 없음'}")
    
    # 1-2. 대응표본 t-검정: 전반 vs 후반 슈팅 수
    # (경기별로 집계된 데이터이므로 여기서는 생략, raw_data 필요)
    
    # 1-3. 카이제곱 검정: 홈/원정 vs 승/패 관계
    contingency_table = pd.crosstab(df['is_home'], df['win'])
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency_table)
    
    results['chi2_home_win'] = {
        'chi2_statistic': chi2,
        'p_value': p_chi,
        'dof': dof,
        'significant': p_chi < 0.05
    }
    
    print(f"\n[카이제곱 검정: 홈/원정 ↔ 승/패 독립성]")
    print(f"  • χ² 통계량: {chi2:.3f}, p-value: {p_chi:.4f}")
    print(f"  → {'홈/원정이 승패에 영향' if p_chi < 0.05 else '홈/원정과 승패는 독립'}")
    
    # 1-4. 정규성 검정 (Shapiro-Wilk)
    stat_sw, p_sw = stats.shapiro(df['success_rate'].sample(min(500, len(df))))
    
    results['normality'] = {
        'statistic': stat_sw,
        'p_value': p_sw,
        'is_normal': p_sw > 0.05
    }
    
    print(f"\n[정규성 검정 (Shapiro-Wilk)]")
    print(f"  • success_rate: W={stat_sw:.4f}, p={p_sw:.4f}")
    print(f"  → {'정규분포 따름' if p_sw > 0.05 else '정규분포 아님'}")
    
    STATS_RESULTS['hypothesis_tests'] = results
    
    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # t-검정 시각화
    ax1 = axes[0]
    data_ttest = [home_success, away_success]
    bp = ax1.boxplot(data_ttest, labels=['홈팀', '원정팀'], patch_artist=True)
    colors = ['lightcoral', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_ylabel('성공률')
    ax1.set_title(f't-검정: 홈 vs 원정\n(p={p_value:.4f})', fontsize=12)
    
    # 카이제곱 시각화
    ax2 = axes[1]
    contingency_table.plot(kind='bar', ax=ax2, color=['lightcoral', 'lightgreen'])
    ax2.set_xlabel('홈/원정 (0=원정, 1=홈)')
    ax2.set_ylabel('빈도')
    ax2.set_title(f'카이제곱: 홈/원정 vs 승패\n(p={p_chi:.4f})', fontsize=12)
    ax2.legend(['패배', '승리'])
    ax2.set_xticklabels(['원정', '홈'], rotation=0)
    
    # 정규성 검정 시각화
    ax3 = axes[2]
    stats.probplot(df['success_rate'], dist="norm", plot=ax3)
    ax3.set_title(f'Q-Q Plot (정규성)\n(p={p_sw:.4f})', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/stats_01_hypothesis_tests.png", dpi=150)
    plt.close()
    print(f"\n🎨 저장: {OUTPUT_PATH}/stats_01_hypothesis_tests.png")
    
    return results


# ============================================
# 2. A/B 테스트
# ============================================
def run_ab_tests(df):
    """A/B 테스트"""
    print_header("STEP 2: A/B 테스트", "🎯")
    
    results = {}
    
    # 2-1. A/B 테스트: 높은 슈팅 (상위 50%) vs 낮은 슈팅 그룹 승률
    median_shots = df['total_shots'].median()
    group_a = df[df['total_shots'] >= median_shots]['win']  # 많은 슈팅
    group_b = df[df['total_shots'] < median_shots]['win']   # 적은 슈팅
    
    # Z-검정 (비율 검정)
    count = np.array([group_a.sum(), group_b.sum()])
    nobs = np.array([len(group_a), len(group_b)])
    z_stat, p_value = proportions_ztest(count, nobs)
    
    results['ab_shots'] = {
        'group_a': '높은 슈팅 그룹',
        'group_b': '낮은 슈팅 그룹',
        'a_win_rate': group_a.mean(),
        'b_win_rate': group_b.mean(),
        'z_statistic': z_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'lift': (group_a.mean() - group_b.mean()) / group_b.mean() * 100 if group_b.mean() > 0 else 0
    }
    
    print(f"\n[A/B 테스트 1: 슈팅 수에 따른 승률]")
    print(f"  • Group A (높은 슈팅) 승률: {group_a.mean():.1%}")
    print(f"  • Group B (낮은 슈팅) 승률: {group_b.mean():.1%}")
    print(f"  • Z-통계량: {z_stat:.3f}, p-value: {p_value:.4f}")
    print(f"  • Lift: {results['ab_shots']['lift']:.1f}%")
    print(f"  → {'슈팅 수가 승률에 유의미한 영향' if p_value < 0.05 else '유의미한 차이 없음'}")
    
    # 2-2. A/B 테스트: 공격적 포지셔닝 vs 수비적 포지셔닝
    median_x = df['avg_x_position'].median()
    group_a2 = df[df['avg_x_position'] >= median_x]['win']  # 공격적
    group_b2 = df[df['avg_x_position'] < median_x]['win']   # 수비적
    
    count2 = np.array([group_a2.sum(), group_b2.sum()])
    nobs2 = np.array([len(group_a2), len(group_b2)])
    z_stat2, p_value2 = proportions_ztest(count2, nobs2)
    
    results['ab_position'] = {
        'group_a': '공격적 포지셔닝',
        'group_b': '수비적 포지셔닝',
        'a_win_rate': group_a2.mean(),
        'b_win_rate': group_b2.mean(),
        'z_statistic': z_stat2,
        'p_value': p_value2,
        'significant': p_value2 < 0.05,
        'lift': (group_a2.mean() - group_b2.mean()) / group_b2.mean() * 100 if group_b2.mean() > 0 else 0
    }
    
    print(f"\n[A/B 테스트 2: 포지셔닝에 따른 승률]")
    print(f"  • Group A (공격적) 승률: {group_a2.mean():.1%}")
    print(f"  • Group B (수비적) 승률: {group_b2.mean():.1%}")
    print(f"  • Z-통계량: {z_stat2:.3f}, p-value: {p_value2:.4f}")
    print(f"  • Lift: {results['ab_position']['lift']:.1f}%")
    
    STATS_RESULTS['ab_tests'] = results
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # A/B 테스트 1
    ax1 = axes[0]
    groups = ['높은 슈팅\n(Group A)', '낮은 슈팅\n(Group B)']
    rates = [group_a.mean(), group_b.mean()]
    colors = ['seagreen' if p_value < 0.05 else 'gray', 'lightcoral']
    bars = ax1.bar(groups, rates, color=colors)
    ax1.set_ylabel('승률')
    ax1.set_title(f'A/B 테스트: 슈팅 수별 승률\n(p={p_value:.4f}, Lift={results["ab_shots"]["lift"]:.1f}%)')
    ax1.set_ylim(0, 1)
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.1%}', ha='center', fontsize=12)
    
    # A/B 테스트 2
    ax2 = axes[1]
    groups2 = ['공격적\n(Group A)', '수비적\n(Group B)']
    rates2 = [group_a2.mean(), group_b2.mean()]
    colors2 = ['seagreen' if p_value2 < 0.05 else 'gray', 'lightcoral']
    bars2 = ax2.bar(groups2, rates2, color=colors2)
    ax2.set_ylabel('승률')
    ax2.set_title(f'A/B 테스트: 포지셔닝별 승률\n(p={p_value2:.4f}, Lift={results["ab_position"]["lift"]:.1f}%)')
    ax2.set_ylim(0, 1)
    for bar, rate in zip(bars2, rates2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.1%}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/stats_02_ab_tests.png", dpi=150)
    plt.close()
    print(f"\n🎨 저장: {OUTPUT_PATH}/stats_02_ab_tests.png")
    
    return results


# ============================================
# 3. 회귀 분석
# ============================================
def run_regression(df):
    """회귀 분석"""
    print_header("STEP 3: 회귀 분석", "📈")
    
    results = {}
    
    # 3-1. 선형 회귀: 슈팅 → 득점 예측
    # (득점 데이터가 팀별로 필요하므로 match_info에서 가져옴)
    df_home = df[df['is_home'] == 1].copy()
    df_home['goals'] = df_home['home_score']
    
    X = df_home[['total_shots', 'success_rate', 'avg_x_position']].fillna(0)
    y = df_home['goals']
    
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    y_pred = lr_model.predict(X)
    
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    results['linear_regression'] = {
        'r2_score': r2,
        'rmse': rmse,
        'coefficients': dict(zip(X.columns, lr_model.coef_)),
        'intercept': lr_model.intercept_
    }
    
    print(f"\n[선형 회귀: 슈팅/성공률/포지션 → 득점]")
    print(f"  • R² Score: {r2:.3f}")
    print(f"  • RMSE: {rmse:.3f}")
    print(f"  • 계수:")
    for col, coef in zip(X.columns, lr_model.coef_):
        print(f"    - {col}: {coef:.4f}")
    
    # 3-2. 로지스틱 회귀: 승리 예측
    X_log = df[['total_shots', 'success_rate', 'avg_x_position', 'total_passes']].fillna(0)
    y_log = df['win']
    
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_log, y_log)
    
    accuracy = log_model.score(X_log, y_log)
    
    results['logistic_regression'] = {
        'accuracy': accuracy,
        'coefficients': dict(zip(X_log.columns, log_model.coef_[0])),
        'odds_ratios': dict(zip(X_log.columns, np.exp(log_model.coef_[0])))
    }
    
    print(f"\n[로지스틱 회귀: 승리 예측]")
    print(f"  • 정확도: {accuracy:.1%}")
    print(f"  • 오즈비 (Odds Ratio):")
    for col, odds in results['logistic_regression']['odds_ratios'].items():
        interpretation = "승리 확률 ↑" if odds > 1 else "승리 확률 ↓"
        print(f"    - {col}: {odds:.3f} ({interpretation})")
    
    STATS_RESULTS['regression'] = results
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 선형 회귀: 실제 vs 예측
    ax1 = axes[0]
    ax1.scatter(y, y_pred, alpha=0.5, color='steelblue')
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    ax1.set_xlabel('실제 득점')
    ax1.set_ylabel('예측 득점')
    ax1.set_title(f'선형 회귀: 실제 vs 예측\n(R²={r2:.3f})', fontsize=12)
    
    # 로지스틱 회귀: 계수 시각화
    ax2 = axes[1]
    coef_df = pd.DataFrame({
        'feature': X_log.columns,
        'coefficient': log_model.coef_[0]
    }).sort_values('coefficient')
    colors = ['lightcoral' if c < 0 else 'seagreen' for c in coef_df['coefficient']]
    ax2.barh(coef_df['feature'], coef_df['coefficient'], color=colors)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('계수 (Coefficient)')
    ax2.set_title(f'로지스틱 회귀 계수\n(정확도={accuracy:.1%})', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/stats_03_regression.png", dpi=150)
    plt.close()
    print(f"\n🎨 저장: {OUTPUT_PATH}/stats_03_regression.png")
    
    return results


# ============================================
# 4. ARIMA 시계열 예측
# ============================================
def run_arima(df):
    """ARIMA 시계열 예측"""
    print_header("STEP 4: ARIMA 시계열 예측", "📉")
    
    results = {}
    
    # 일별 평균 득점 시계열 생성
    df_sorted = df.sort_values('game_date')
    daily_stats = df_sorted.groupby('game_date').agg({
        'total_actions': 'mean',
        'success_rate': 'mean',
        'total_shots': 'mean'
    }).reset_index()
    
    # 시계열 인덱스 설정
    daily_stats.set_index('game_date', inplace=True)
    daily_stats = daily_stats.asfreq('D', method='ffill')  # 결측일 채우기
    
    # ARIMA 모델 (success_rate 예측)
    try:
        ts = daily_stats['success_rate'].dropna()
        
        if len(ts) >= 30:
            # ARIMA(1,1,1) 모델
            model = ARIMA(ts, order=(1, 1, 1))
            model_fit = model.fit()
            
            # 7일 예측
            forecast = model_fit.forecast(steps=7)
            
            results['arima'] = {
                'aic': model_fit.aic,
                'bic': model_fit.bic,
                'forecast_7days': forecast.tolist(),
                'forecast_mean': forecast.mean()
            }
            
            print(f"\n[ARIMA(1,1,1): 성공률 예측]")
            print(f"  • AIC: {model_fit.aic:.2f}")
            print(f"  • BIC: {model_fit.bic:.2f}")
            print(f"  • 향후 7일 예측 평균: {forecast.mean():.3f}")
            
            # 시각화
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # 실제 시계열 + 예측
            ax1 = axes[0]
            ax1.plot(ts.index, ts.values, label='실제 성공률', color='steelblue')
            forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=7)
            ax1.plot(forecast_index, forecast, label='ARIMA 예측', color='coral', linestyle='--', marker='o')
            ax1.set_ylabel('성공률')
            ax1.set_title('ARIMA 시계열 예측', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 잔차 분석
            ax2 = axes[1]
            residuals = model_fit.resid
            ax2.plot(residuals, color='gray', alpha=0.7)
            ax2.axhline(y=0, color='red', linestyle='--')
            ax2.set_ylabel('잔차')
            ax2.set_title('ARIMA 잔차 분석', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_PATH}/stats_04_arima.png", dpi=150)
            plt.close()
            print(f"🎨 저장: {OUTPUT_PATH}/stats_04_arima.png")
        else:
            print("⚠️ 데이터 부족으로 ARIMA 분석 생략")
            results['arima'] = {'error': 'Insufficient data'}
            
    except Exception as e:
        print(f"⚠️ ARIMA 오류: {e}")
        results['arima'] = {'error': str(e)}
    
    STATS_RESULTS['arima'] = results
    
    return results


# ============================================
# 5. 시계열 분해 (STL Decomposition)
# ============================================
def run_timeseries_decomposition(df):
    """시계열 분해"""
    print_header("STEP 5: 시계열 분해 (Trend, Seasonality, Residual)", "🔄")
    
    results = {}
    
    # 일별 집계
    df_sorted = df.sort_values('game_date')
    daily_stats = df_sorted.groupby('game_date').agg({
        'total_actions': 'mean'
    }).reset_index()
    
    daily_stats.set_index('game_date', inplace=True)
    daily_stats = daily_stats.asfreq('D', method='ffill')
    
    try:
        ts = daily_stats['total_actions'].dropna()
        
        if len(ts) >= 14:
            # STL 분해 (주기=7)
            decomposition = seasonal_decompose(ts, model='additive', period=7)
            
            results['decomposition'] = {
                'trend_mean': decomposition.trend.dropna().mean(),
                'seasonal_amplitude': decomposition.seasonal.max() - decomposition.seasonal.min(),
                'residual_std': decomposition.resid.dropna().std()
            }
            
            print(f"\n[시계열 분해 결과]")
            print(f"  • Trend 평균: {results['decomposition']['trend_mean']:.2f}")
            print(f"  • Seasonal 진폭: {results['decomposition']['seasonal_amplitude']:.2f}")
            print(f"  • Residual 표준편차: {results['decomposition']['residual_std']:.2f}")
            
            # 시각화
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            
            decomposition.observed.plot(ax=axes[0], title='원본 시계열', color='steelblue')
            decomposition.trend.plot(ax=axes[1], title='Trend (추세)', color='coral')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonality (계절성)', color='seagreen')
            decomposition.resid.plot(ax=axes[3], title='Residual (잔차)', color='gray')
            
            for ax in axes:
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_PATH}/stats_05_decomposition.png", dpi=150)
            plt.close()
            print(f"🎨 저장: {OUTPUT_PATH}/stats_05_decomposition.png")
        else:
            print("⚠️ 데이터 부족으로 시계열 분해 생략")
            results['decomposition'] = {'error': 'Insufficient data'}
            
    except Exception as e:
        print(f"⚠️ 시계열 분해 오류: {e}")
        results['decomposition'] = {'error': str(e)}
    
    STATS_RESULTS['decomposition'] = results
    
    return results


# ============================================
# 6. 인사이트 요약
# ============================================
def generate_stats_insights():
    """일반 통계 분석 인사이트"""
    print_header("FINAL: 일반 통계 분석 인사이트", "💡")
    
    insights = []
    
    # 가설검정 인사이트
    if 'hypothesis_tests' in STATS_RESULTS:
        ht = STATS_RESULTS['hypothesis_tests']
        if ht['chi2_home_win']['significant']:
            insights.append("✅ 홈/원정 여부가 승패에 통계적으로 유의미한 영향을 미침")
        else:
            insights.append("ℹ️ 홈/원정과 승패는 독립적 (K-리그 홈 어드밴티지 약함)")
    
    # A/B 테스트 인사이트
    if 'ab_tests' in STATS_RESULTS:
        ab = STATS_RESULTS['ab_tests']
        if ab['ab_shots']['significant']:
            lift = ab['ab_shots']['lift']
            insights.append(f"✅ 슈팅 수가 많은 팀이 승률 {abs(lift):.1f}% {'높음' if lift > 0 else '낮음'}")
    
    # 회귀 인사이트
    if 'regression' in STATS_RESULTS:
        reg = STATS_RESULTS['regression']
        insights.append(f"📈 선형 회귀 R²={reg['linear_regression']['r2_score']:.3f} (득점 예측력)")
        
        # 가장 영향력 있는 변수 찾기
        odds = reg['logistic_regression']['odds_ratios']
        max_var = max(odds, key=lambda x: abs(odds[x] - 1))
        insights.append(f"🎯 승리에 가장 큰 영향: '{max_var}' (OR={odds[max_var]:.2f})")
    
    print("\n📋 [일반 통계 분석 핵심 결론]")
    for insight in insights:
        print(f"   {insight}")
    
    # 리포트 저장
    with open(f"{OUTPUT_PATH}/stats_insights.txt", "w", encoding="utf-8") as f:
        f.write("K-리그 일반 통계 분석 리포트\n")
        f.write("=" * 50 + "\n\n")
        f.write("[분석 항목]\n")
        f.write("1. 가설검정 (t-검정, 카이제곱)\n")
        f.write("2. A/B 테스트\n")
        f.write("3. 회귀 분석 (선형/로지스틱)\n")
        f.write("4. ARIMA 시계열 예측\n")
        f.write("5. 시계열 분해\n\n")
        f.write("[인사이트]\n")
        for insight in insights:
            f.write(insight + "\n")
    
    print(f"\n📄 저장: {OUTPUT_PATH}/stats_insights.txt")
    
    return insights


# ============================================
# MAIN
# ============================================
def main():
    print("\n" + "📊" * 25)
    print("   K-리그 일반 통계 분석 시스템")
    print("   [가설검정 → A/B테스트 → 회귀 → ARIMA → 분해]")
    print("📊" * 25 + "\n")
    
    try:
        match_info, raw_data, df = load_data()
        
        run_hypothesis_tests(df)
        run_ab_tests(df)
        run_regression(df)
        run_arima(df)
        run_timeseries_decomposition(df)
        
        generate_stats_insights()
        
        print("\n" + "✅" * 25)
        print("   일반 통계 분석 완료!")
        print("✅" * 25)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
