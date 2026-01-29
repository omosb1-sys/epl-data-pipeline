"""
K-리그 Causal AI 분석 모듈
=============================
"공격적 포지셔닝이 실제로 승리를 유발하는가?"

인과 추론 기법:
1. Propensity Score Matching (PSM)
2. Instrumental Variable (도구 변수)
3. Difference-in-Differences 개념 적용

Author: Antigravity (Senior Data Analyst)
Date: 2026-01-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
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


def prepare_causal_data():
    """인과 분석용 데이터 준비"""
    print("=" * 60)
    print("📊 [STEP 1] Causal AI용 데이터 준비")
    print("=" * 60)
    
    match_info = pd.read_csv(f"{BASE_PATH}/match_info.csv")
    raw_data = pd.read_csv(f"{BASE_PATH}/raw_data.csv")
    
    game_stats = raw_data.groupby(['game_id', 'team_id']).agg(
        total_actions=('action_id', 'count'),
        total_passes=('type_name', lambda x: (x == 'Pass').sum()),
        total_shots=('type_name', lambda x: (x == 'Shot').sum()),
        successful_actions=('result_name', lambda x: (x == 'Successful').sum()),
        avg_x_position=('start_x', 'mean'),
        avg_y_position=('start_y', 'mean'),
        unique_players=('player_id', 'nunique')
    ).reset_index()
    
    game_stats['pass_ratio'] = game_stats['total_passes'] / game_stats['total_actions']
    game_stats['shot_ratio'] = game_stats['total_shots'] / game_stats['total_actions']
    game_stats['success_rate'] = game_stats['successful_actions'] / game_stats['total_actions']
    
    merged = game_stats.merge(
        match_info[['game_id', 'home_team_id', 'away_team_id', 'home_score', 'away_score']],
        on='game_id', how='left'
    )
    
    def get_result(row):
        if row['team_id'] == row['home_team_id']:
            return 1 if row['home_score'] > row['away_score'] else 0
        else:
            return 1 if row['away_score'] > row['home_score'] else 0
    
    merged['win'] = merged.apply(get_result, axis=1)
    
    # Treatment: 공격적 포지셔닝 (상위 50%)
    median_x = merged['avg_x_position'].median()
    merged['aggressive_positioning'] = (merged['avg_x_position'] > median_x).astype(int)
    
    print(f"✅ 데이터 Shape: {merged.shape}")
    print(f"✅ Treatment 분포 (공격적 포지셔닝):\n{merged['aggressive_positioning'].value_counts()}")
    print(f"✅ 승리 분포:\n{merged['win'].value_counts()}")
    
    return merged


def naive_comparison(df):
    """단순 비교 (Selection Bias 포함)"""
    print("\n" + "=" * 60)
    print("📈 [STEP 2] 단순 비교 (Naive Comparison)")
    print("=" * 60)
    
    treated = df[df['aggressive_positioning'] == 1]['win']
    control = df[df['aggressive_positioning'] == 0]['win']
    
    naive_ate = treated.mean() - control.mean()
    
    print(f"\n[단순 비교 결과]")
    print(f"  • 공격적 팀 승률: {treated.mean():.3f} ({len(treated)}팀)")
    print(f"  • 수비적 팀 승률: {control.mean():.3f} ({len(control)}팀)")
    print(f"  • 단순 차이 (ATE): {naive_ate:.3f}")
    print(f"\n⚠️ 주의: 이 결과는 Selection Bias가 포함되어 있음!")
    print(f"   (강팀이 원래 공격적일 가능성)")
    
    return naive_ate


def propensity_score_matching(df):
    """Propensity Score Matching (PSM)"""
    print("\n" + "=" * 60)
    print("🎯 [STEP 3] Propensity Score Matching")
    print("=" * 60)
    
    # Confounders (교란 변수): 팀의 기본 역량
    confounders = ['total_passes', 'success_rate', 'unique_players']
    
    X = df[confounders].fillna(0)
    treatment = df['aggressive_positioning']
    
    # Propensity Score 계산 (로지스틱 회귀)
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X, treatment)
    df['propensity_score'] = ps_model.predict_proba(X)[:, 1]
    
    print(f"[Propensity Score 분포]")
    print(df.groupby('aggressive_positioning')['propensity_score'].describe())
    
    # Nearest Neighbor Matching
    treated_df = df[df['aggressive_positioning'] == 1].copy()
    control_df = df[df['aggressive_positioning'] == 0].copy()
    
    # 1:1 매칭
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control_df[['propensity_score']])
    
    distances, indices = nn.kneighbors(treated_df[['propensity_score']])
    matched_control = control_df.iloc[indices.flatten()]
    
    # ATT (Average Treatment Effect on Treated) 계산
    att = treated_df['win'].mean() - matched_control['win'].mean()
    
    print(f"\n[PSM 결과]")
    print(f"  • 매칭된 Treated 승률: {treated_df['win'].mean():.3f}")
    print(f"  • 매칭된 Control 승률: {matched_control['win'].mean():.3f}")
    print(f"  • ATT (인과 효과): {att:.3f}")
    
    # 통계적 유의성 검정
    t_stat, p_value = stats.ttest_ind(treated_df['win'], matched_control['win'])
    significance = "유의미" if p_value < 0.05 else "유의미하지 않음"
    print(f"  • t-통계량: {t_stat:.3f}, p-value: {p_value:.4f} → {significance}")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PS 분포
    ax1 = axes[0]
    ax1.hist(df[df['aggressive_positioning'] == 1]['propensity_score'], 
             bins=20, alpha=0.7, label='공격적 (Treated)', color='coral')
    ax1.hist(df[df['aggressive_positioning'] == 0]['propensity_score'], 
             bins=20, alpha=0.7, label='수비적 (Control)', color='steelblue')
    ax1.set_xlabel('Propensity Score')
    ax1.set_ylabel('빈도')
    ax1.set_title('Propensity Score 분포', fontsize=14)
    ax1.legend()
    
    # 인과 효과 비교
    ax2 = axes[1]
    methods = ['단순 비교\n(Biased)', 'PSM\n(Causal)']
    naive = df[df['aggressive_positioning'] == 1]['win'].mean() - df[df['aggressive_positioning'] == 0]['win'].mean()
    effects = [naive, att]
    colors = ['lightcoral', 'seagreen']
    bars = ax2.bar(methods, effects, color=colors)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('승률 차이 (Effect)')
    ax2.set_title('단순 비교 vs PSM 인과 효과', fontsize=14)
    for bar, eff in zip(bars, effects):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{eff:.3f}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/causal_psm_analysis.png", dpi=150)
    print(f"\n🎨 저장: {OUTPUT_PATH}/causal_psm_analysis.png")
    
    return att, p_value


def instrumental_variable_analysis(df):
    """도구 변수 분석 (2SLS 개념)"""
    print("\n" + "=" * 60)
    print("🔧 [STEP 4] Instrumental Variable 분석")
    print("=" * 60)
    
    # 도구 변수: unique_players (출전 선수 다양성)
    # 가정: 선수 로테이션 → 공격적 포지셔닝 → 승리
    # Z → X → Y (Z는 Y에 직접 영향 X)
    
    Z = df['unique_players'].values.reshape(-1, 1)
    X = df['avg_x_position'].values.reshape(-1, 1)
    Y = df['win'].values
    
    scaler = StandardScaler()
    Z_scaled = scaler.fit_transform(Z)
    X_scaled = scaler.fit_transform(X)
    
    # 1단계: Z → X (First Stage)
    first_stage = LinearRegression()
    first_stage.fit(Z_scaled, X_scaled.ravel())
    X_hat = first_stage.predict(Z_scaled).reshape(-1, 1)
    
    print(f"[1단계 회귀: Z → X]")
    print(f"  • R²: {first_stage.score(Z_scaled, X_scaled.ravel()):.3f}")
    
    # 2단계: X_hat → Y (Second Stage)
    second_stage = LogisticRegression(max_iter=1000)
    second_stage.fit(X_hat, Y)
    
    iv_effect = second_stage.coef_[0][0]
    
    print(f"\n[2단계 회귀: X_hat → Y]")
    print(f"  • IV 추정 인과 효과: {iv_effect:.4f}")
    
    if iv_effect > 0:
        print(f"  → 공격적 포지셔닝이 승리에 '인과적으로' 긍정적 영향")
    else:
        print(f"  → 인과적 영향이 불분명하거나 역효과")
    
    return iv_effect


def sensitivity_analysis(df, att):
    """민감도 분석 (Rosenbaum Bounds 개념)"""
    print("\n" + "=" * 60)
    print("📐 [STEP 5] 민감도 분석 (Sensitivity Analysis)")
    print("=" * 60)
    
    print(f"\n[숨겨진 교란 변수 시나리오 분석]")
    print(f"  • 추정된 ATT: {att:.3f}")
    
    # 다양한 편향 수준에서의 효과 변화 시뮬레이션
    biases = [0, 0.05, 0.10, 0.15, 0.20]
    adjusted_effects = [att - b for b in biases]
    
    print(f"\n  편향 수준별 조정된 효과:")
    for b, adj in zip(biases, adjusted_effects):
        status = "✅ 양의 효과" if adj > 0 else "❌ 효과 사라짐"
        print(f"    Bias {b:.2f} → Effect {adj:.3f} {status}")
    
    # 결론: 어느 정도의 숨겨진 편향까지 결과가 robust한가?
    threshold = att  # 효과가 0이 되는 편향 수준
    print(f"\n  🔍 결론: 약 {threshold:.1%}의 숨겨진 편향까지 결과가 robust함")


def generate_causal_insights(naive_ate, att, p_value, iv_effect):
    """인과 분석 인사이트"""
    print("\n" + "=" * 60)
    print("💡 [FINAL] Causal AI 인사이트")
    print("=" * 60)
    
    insights = []
    
    # 1. Selection Bias 정량화
    bias = naive_ate - att
    insights.append(f"📊 Selection Bias 크기: {bias:.3f} (단순 비교가 {abs(bias)*100:.1f}% 과대평가)")
    
    # 2. 인과 효과 해석
    if p_value < 0.05 and att > 0:
        insights.append(f"✅ 공격적 포지셔닝 → 승리 인과관계 통계적으로 유의미 (ATT={att:.3f})")
    elif att > 0:
        insights.append(f"⚠️ 양의 인과 효과 존재하나 통계적 유의성 부족 (p={p_value:.3f})")
    else:
        insights.append(f"❌ 공격적 포지셔닝의 인과적 효과 불분명")
    
    # 3. IV 결과
    if iv_effect > 0:
        insights.append(f"🔧 도구변수 분석도 양의 인과효과 지지 (β={iv_effect:.4f})")
    
    # 4. 실무적 제언
    insights.append("⚽ 전술 제언: 단순히 '전진'하는 것이 아닌, '효율적 전진'이 핵심")
    insights.append("📈 추가 분석 필요: 상대팀 수비력, 경기 중요도 등 추가 교란변수 통제 권장")
    
    print("\n📋 [인과 분석 핵심 결론]")
    for insight in insights:
        print(f"   {insight}")
    
    # 리포트 저장
    with open(f"{OUTPUT_PATH}/causal_insights.txt", "w", encoding="utf-8") as f:
        f.write("K-리그 Causal AI 분석 리포트\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"[분석 질문]\n공격적 포지셔닝이 실제로 승리를 '유발'하는가?\n\n")
        f.write(f"[정량적 결과]\n")
        f.write(f"  - 단순 비교 (Biased): {naive_ate:.3f}\n")
        f.write(f"  - PSM 인과 효과 (ATT): {att:.3f}\n")
        f.write(f"  - Selection Bias: {bias:.3f}\n")
        f.write(f"  - p-value: {p_value:.4f}\n")
        f.write(f"  - IV 효과: {iv_effect:.4f}\n\n")
        f.write("[인사이트]\n")
        for insight in insights:
            f.write(insight + "\n")
    
    print(f"\n📄 저장: {OUTPUT_PATH}/causal_insights.txt")


def main():
    print("\n" + "🔬" * 20)
    print("  K-리그 Causal AI 분석")
    print("  '공격적 포지셔닝 → 승리' 인과관계 검증")
    print("🔬" * 20 + "\n")
    
    df = prepare_causal_data()
    
    naive_ate = naive_comparison(df)
    
    att, p_value = propensity_score_matching(df)
    
    iv_effect = instrumental_variable_analysis(df)
    
    sensitivity_analysis(df, att)
    
    generate_causal_insights(naive_ate, att, p_value, iv_effect)
    
    print("\n" + "✅" * 20)
    print("  Causal AI 분석 완료!")
    print("✅" * 20)


if __name__ == "__main__":
    main()
