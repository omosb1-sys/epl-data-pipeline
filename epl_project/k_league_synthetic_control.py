"""
K-리그 Synthetic Control Method (SCM) 분석 모듈
================================================
LinkedIn 인사이트 적용: "언제 A/B 테스트나 Diff-in-Diff로 부족한가?"

주요 기능:
1. Synthetic Twin 생성: Donor units 간의 최적 가중치 산출 (Ridge Regression)
2. 인과 효과 계산: 실제값 - Synthetic Control값
3. Placebo Test: Permutation test를 통한 empirical p-value 도출
4. 시각화: Pre-trend 매칭 및 Post-intervention Gap 확인

Author: Antigravity (Senior Data Analyst)
Date: 2026-01-23
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings
import os

warnings.filterwarnings('ignore')

# 환경 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

class SyntheticControl:
    """Synthetic Control Method 분석 클래스"""
    
    def __init__(self, df: pd.DataFrame, time_col: str, unit_col: str, target_col: str):
        self.df = df
        self.time_col = time_col
        self.unit_col = unit_col
        self.target_col = target_col
        self.weights = None
        self.donor_units = None
        self.treated_unit = None
        self.intervention_time = None
        
    def fit(self, treated_unit: str, donor_units: list, intervention_time: int):
        """Synthetic Twin의 가중치를 학습 (Pre-intervention 기간 기준)"""
        self.treated_unit = treated_unit
        self.donor_units = donor_units
        self.intervention_time = intervention_time
        
        # Pre-intervention 데이터 준비
        pre_df = self.df[self.df[self.time_col] < intervention_time]
        
        # Matrix form으로 변환
        y_treated = pre_df[pre_df[self.unit_col] == treated_unit][self.target_col].values
        
        X_donors = []
        for donor in donor_units:
            X_donors.append(pre_df[pre_df[self.unit_col] == donor][self.target_col].values)
        
        X_donors = np.array(X_donors).T # (time, donors)
        
        # Ridge Regression을 사용하여 가중치 계산 (ML 기반 가중치)
        # alpha=0.1로 규제 적용하여 overfitting 방지
        model = Ridge(alpha=0.1, fit_intercept=False, positive=True)
        model.fit(X_donors, y_treated)
        
        self.weights = model.coef_
        # 가중치 합이 1이 되도록 정규화 (전통적 SCM 방식)
        if self.weights.sum() > 0:
            self.weights = self.weights / self.weights.sum()
            
        print(f"✅ Synthetic Control 가중치 학습 완료 ({treated_unit})")
        for donor, weight in zip(donor_units, self.weights):
            if weight > 0.01:
                print(f"  • {donor}: {weight:.3f}")
                
    def predict(self) -> pd.DataFrame:
        """학습된 가중치로 전 기간의 Synthetic Control 값을 생성"""
        results = []
        all_times = sorted(self.df[self.time_col].unique())
        
        for t in all_times:
            # Check if treated unit exists for this time
            treated_rows = self.df[(self.df[self.unit_col] == self.treated_unit) & (self.df[self.time_col] == t)]
            if len(treated_rows) == 0:
                continue
            actual = treated_rows[self.target_col].values[0]
            
            synthetic = 0
            for donor, weight in zip(self.donor_units, self.weights):
                donor_rows = self.df[(self.df[self.unit_col] == donor) & (self.df[self.time_col] == t)]
                val = donor_rows[self.target_col].values[0] if len(donor_rows) > 0 else 0
                synthetic += val * weight
                
            results.append({
                'time': t,
                'actual': actual,
                'synthetic': synthetic,
                'gap': actual - synthetic
            })
            
        return pd.DataFrame(results)

    def run_placebo_test(self, n_placebos: int = 10) -> float:
        """Placebo (Permutation) Test를 통한 p-value 계산"""
        print(f"🧪 Placebo Test 수행 중 (n={n_placebos})...")
        
        real_results = self.predict()
        real_effect = real_results[real_results['time'] >= self.intervention_time]['gap'].mean()
        
        placebo_effects = []
        # Donor unit들 중 일부를 placebo treated unit으로 가정
        test_units = self.donor_units[:min(len(self.donor_units), n_placebos)]
        
        for p_unit in test_units:
            other_donors = [u for u in self.donor_units if u != p_unit] + [self.treated_unit]
            p_scm = SyntheticControl(self.df, self.time_col, self.unit_col, self.target_col)
            try:
                p_scm.fit(p_unit, other_donors, self.intervention_time)
                p_res = p_scm.predict()
                p_effect = p_res[p_res['time'] >= self.intervention_time]['gap'].mean()
                placebo_effects.append(abs(p_effect))
            except Exception as e:
                continue
                
        # Empirical p-value: 실제 효과보다 큰 placebo 효과의 비율
        p_value = np.mean([1 if pe >= abs(real_effect) else 0 for pe in placebo_effects]) if len(placebo_effects) > 0 else 1.0
        return p_value, placebo_effects

    def visualize(self, results: pd.DataFrame, treated_name: str, p_value: float = None):
        """분석 결과 시각화"""
        plt.figure(figsize=(12, 6))
        plt.plot(results['time'], results['actual'], 'o-', label=f'Actual ({treated_name})', color='crimson', linewidth=2)
        plt.plot(results['time'], results['synthetic'], '--', label='Synthetic Control', color='black', alpha=0.7)
        
        plt.axvline(x=self.intervention_time, color='gray', linestyle=':', label='Intervention')
        
        plt.title(f"Synthetic Control Analysis: {treated_name}", fontsize=15)
        if p_value is not None:
            plt.suptitle(f"Empirical p-value: {p_value:.3f}", y=0.92, fontsize=12)
            
        plt.xlabel("Time (Round/Week)")
        plt.ylabel(self.target_col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = f"{OUTPUT_PATH}/scm_analysis_{treated_name}.png"
        plt.savefig(save_path, dpi=150)
        print(f"🎨 시각화 저장 완료: {save_path}")
        plt.close()

def simulate_data():
    """테스트를 위한 시뮬레이션 데이터 생성"""
    np.random.seed(42)
    rounds = np.arange(1, 21)
    teams = ['울산', '전북', '포항', '광주', '대구', '서울']
    
    data = []
    # Baseline trend
    base_trend = np.linspace(1.2, 1.8, 20)
    
    for team in teams:
        noise = np.random.normal(0, 0.1, 20)
        offset = np.random.uniform(-0.2, 0.2)
        vals = base_trend + offset + noise
        
        # '울산'에 11라운드부터 효과(득점 상승) 주입
        if team == '울산':
            vals[10:] += 0.5
            
        for r, v in zip(rounds, vals):
            data.append({'round': r, 'team': team, 'goals': max(0, v)})
            
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("🚀 Synthetic Control Method 시뮬레이션 시작")
    df = simulate_data()
    
    intervention_round = 11
    treated = '울산'
    donors = ['전북', '포항', '광주', '대구', '서울']
    
    scm = SyntheticControl(df, 'round', 'team', 'goals')
    scm.fit(treated, donors, intervention_round)
    
    results = scm.predict()
    p_val, placebos = scm.run_placebo_test(n_placebos=5)
    
    print(f"\n[분석 결과]")
    post_actual = results[results['time'] >= intervention_round]['actual'].mean()
    post_synth = results[results['time'] >= intervention_round]['synthetic'].mean()
    print(f"  • 실제 평균 득점 (Post): {post_actual:.3f}")
    print(f"  • Synthetic 평균 득점 (Post): {post_synth:.3f}")
    print(f"  • 추정 효과 (Gap): {post_actual - post_synth:.3f}")
    print(f"  • p-value: {p_val:.3f}")
    
    scm.visualize(results, treated, p_val)
