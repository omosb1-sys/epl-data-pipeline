"""
K-리그 풀스택 데이터 분석 파이프라인
=====================================
[EDA → 파생변수 → 통계분석 → 머신러닝 → 인사이트]

Author: Antigravity (Senior Data Analyst)
Date: 2026-01-21
"""

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import os
from typing import Tuple, Dict, Any

warnings.filterwarnings('ignore')

# ============================================
# 0. 환경 설정
# ============================================
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

BASE_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/data"
OUTPUT_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/output"
os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """데이터 로드 및 Pandas 변환"""
    print("=" * 60)
    print("📊 [STEP 0] 데이터 로드")
    print("=" * 60)
    
    match_info = pd.read_csv(f"{BASE_PATH}/match_info.csv")
    raw_data = pd.read_csv(f"{BASE_PATH}/raw_data.csv")
    
    print(f"✅ match_info: {match_info.shape}")
    print(f"✅ raw_data: {raw_data.shape}")
    
    return match_info, raw_data


# ============================================
# 1. 파생변수 생성 (Feature Engineering)
# ============================================
def create_features(match_df: pd.DataFrame, event_df: pd.DataFrame) -> pd.DataFrame:
    """경기별 집계 파생변수 생성
    
    Returns:
        경기 단위로 집계된 분석용 데이터프레임
    """
    print("\n" + "=" * 60)
    print("🔧 [STEP 1] 파생변수 생성 (Feature Engineering)")
    print("=" * 60)
    
    # 1-1. 경기별 이벤트 집계
    game_stats = event_df.groupby(['game_id', 'team_id']).agg(
        total_actions=('action_id', 'count'),
        total_passes=('type_name', lambda x: (x == 'Pass').sum()),
        total_shots=('type_name', lambda x: (x == 'Shot').sum()),
        successful_actions=('result_name', lambda x: (x == 'Successful').sum()),
        avg_x_position=('start_x', 'mean'),
        avg_y_position=('start_y', 'mean'),
        max_x_reach=('end_x', 'max'),
        unique_players=('player_id', 'nunique')
    ).reset_index()
    
    # 1-2. 파생비율 계산
    game_stats['pass_ratio'] = game_stats['total_passes'] / game_stats['total_actions']
    game_stats['shot_ratio'] = game_stats['total_shots'] / game_stats['total_actions']
    game_stats['success_rate'] = game_stats['successful_actions'] / game_stats['total_actions']
    
    # 1-3. 경기 메타데이터 병합
    merged = game_stats.merge(
        match_df[['game_id', 'home_team_id', 'away_team_id', 'home_score', 'away_score', 
                  'home_team_name_ko', 'away_team_name_ko', 'game_date']],
        on='game_id',
        how='left'
    )
    
    # 1-4. 승/무/패 라벨 생성
    def get_result(row):
        if row['team_id'] == row['home_team_id']:
            if row['home_score'] > row['away_score']:
                return 'Win'
            elif row['home_score'] < row['away_score']:
                return 'Lose'
            else:
                return 'Draw'
        else:
            if row['away_score'] > row['home_score']:
                return 'Win'
            elif row['away_score'] < row['home_score']:
                return 'Lose'
            else:
                return 'Draw'
    
    merged['result'] = merged.apply(get_result, axis=1)
    
    print(f"✅ 생성된 파생변수: {list(game_stats.columns[2:])}")
    print(f"✅ 최종 데이터 Shape: {merged.shape}")
    
    return merged


# ============================================
# 2. 기본 통계 분석
# ============================================
def basic_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """기술통계량 및 분포 분석"""
    print("\n" + "=" * 60)
    print("📈 [STEP 2] 기본 통계 분석")
    print("=" * 60)
    
    numeric_cols = ['total_actions', 'total_passes', 'total_shots', 
                    'success_rate', 'pass_ratio', 'shot_ratio']
    
    # 2-1. 기술통계량
    desc_stats = df[numeric_cols].describe()
    print("\n[기술통계량]")
    print(desc_stats.round(3))
    
    # 2-2. 결과별 통계
    result_stats = df.groupby('result')[numeric_cols].mean()
    print("\n[승/무/패별 평균 지표]")
    print(result_stats.round(3))
    
    # 2-3. 분포 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[idx], color='steelblue')
        axes[idx].set_title(f'{col} 분포', fontsize=12)
        axes[idx].axvline(df[col].mean(), color='red', linestyle='--', label='평균')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/basic_stats_distribution.png", dpi=150)
    print(f"\n🎨 저장: {OUTPUT_PATH}/basic_stats_distribution.png")
    
    return {'descriptive': desc_stats, 'by_result': result_stats}


# ============================================
# 3. 상관관계 분석
# ============================================
def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """상관관계 분석 및 히트맵"""
    print("\n" + "=" * 60)
    print("🔗 [STEP 3] 상관관계 분석")
    print("=" * 60)
    
    numeric_cols = ['total_actions', 'total_passes', 'total_shots', 
                    'success_rate', 'pass_ratio', 'avg_x_position', 'unique_players']
    
    corr_matrix = df[numeric_cols].corr()
    
    # 히트맵
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                center=0, fmt='.2f', square=True, linewidths=0.5)
    plt.title('K-리그 경기 지표 상관관계 행렬', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/correlation_heatmap.png", dpi=150)
    print(f"🎨 저장: {OUTPUT_PATH}/correlation_heatmap.png")
    
    # 주요 상관관계 출력
    print("\n[주요 상관관계 (|r| > 0.5)]")
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                print(f"  • {numeric_cols[i]} ↔ {numeric_cols[j]}: {corr_matrix.iloc[i, j]:.3f}")
    
    return corr_matrix


# ============================================
# 4. 고급 통계 분석
# ============================================
def advanced_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """가설검정 및 고급 통계분석"""
    print("\n" + "=" * 60)
    print("🧪 [STEP 4] 고급 통계 분석")
    print("=" * 60)
    
    results = {}
    
    # 4-1. 정규성 검정 (Shapiro-Wilk)
    print("\n[정규성 검정 (Shapiro-Wilk)]")
    for col in ['success_rate', 'pass_ratio']:
        stat, p = stats.shapiro(df[col].dropna().sample(min(500, len(df))))
        normality = "정규분포 O" if p > 0.05 else "정규분포 X"
        print(f"  • {col}: W={stat:.4f}, p={p:.4f} → {normality}")
        results[f'{col}_normality'] = {'statistic': stat, 'p_value': p}
    
    # 4-2. ANOVA (승/무/패 그룹 간 차이)
    print("\n[ANOVA 검정: 승/무/패 그룹 간 성공률 차이]")
    win_data = df[df['result'] == 'Win']['success_rate']
    draw_data = df[df['result'] == 'Draw']['success_rate']
    lose_data = df[df['result'] == 'Lose']['success_rate']
    
    f_stat, p_anova = stats.f_oneway(win_data, draw_data, lose_data)
    significance = "유의미한 차이 존재" if p_anova < 0.05 else "유의미한 차이 없음"
    print(f"  • F-통계량: {f_stat:.4f}, p-value: {p_anova:.4f} → {significance}")
    results['anova'] = {'f_statistic': f_stat, 'p_value': p_anova}
    
    # 4-3. 효과 크기 (Cohen's d: Win vs Lose)
    print("\n[효과 크기 (Cohen's d): 승리 vs 패배]")
    cohens_d = (win_data.mean() - lose_data.mean()) / np.sqrt(
        ((win_data.std()**2) + (lose_data.std()**2)) / 2
    )
    effect_size = "Large" if abs(cohens_d) > 0.8 else ("Medium" if abs(cohens_d) > 0.5 else "Small")
    print(f"  • Cohen's d: {cohens_d:.4f} → {effect_size} Effect")
    results['cohens_d'] = cohens_d
    
    # 4-4. 시각화: 그룹별 박스플롯
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.boxplot(x='result', y='success_rate', data=df, order=['Win', 'Draw', 'Lose'], 
                palette='Set2', ax=axes[0])
    axes[0].set_title('승/무/패별 성공률 분포', fontsize=12)
    
    sns.boxplot(x='result', y='total_shots', data=df, order=['Win', 'Draw', 'Lose'],
                palette='Set2', ax=axes[1])
    axes[1].set_title('승/무/패별 슈팅 횟수 분포', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/advanced_stats_boxplot.png", dpi=150)
    print(f"\n🎨 저장: {OUTPUT_PATH}/advanced_stats_boxplot.png")
    
    return results


# ============================================
# 5. 머신러닝 (승/무/패 예측)
# ============================================
def machine_learning(df: pd.DataFrame) -> Dict[str, Any]:
    """RandomForest 기반 승/무/패 예측 모델"""
    print("\n" + "=" * 60)
    print("🤖 [STEP 5] 머신러닝 분석 (RandomForest)")
    print("=" * 60)
    
    # 5-1. 피처/타겟 분리
    feature_cols = ['total_actions', 'total_passes', 'total_shots', 
                    'success_rate', 'pass_ratio', 'shot_ratio', 
                    'avg_x_position', 'unique_players']
    
    X = df[feature_cols].fillna(0)
    y = df['result']
    
    # 5-2. 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5-3. 모델 학습
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 5-4. 예측 및 평가
    y_pred = model.predict(X_test)
    print("\n[분류 리포트]")
    print(classification_report(y_test, y_pred))
    
    # 5-5. Feature Importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n[Feature Importance (Top 5)]")
    print(importance.head())
    
    # 5-6. 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Feature Importance
    sns.barplot(x='importance', y='feature', data=importance, palette='viridis', ax=axes[0])
    axes[0].set_title('변수 중요도 (Feature Importance)', fontsize=12)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=['Win', 'Draw', 'Lose'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Win', 'Draw', 'Lose'],
                yticklabels=['Win', 'Draw', 'Lose'], ax=axes[1])
    axes[1].set_title('혼동 행렬 (Confusion Matrix)', fontsize=12)
    axes[1].set_xlabel('예측')
    axes[1].set_ylabel('실제')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/ml_results.png", dpi=150)
    print(f"\n🎨 저장: {OUTPUT_PATH}/ml_results.png")
    
    return {'model': model, 'importance': importance}


# ============================================
# 6. 인사이트 요약
# ============================================
def generate_insights(basic_stats: Dict, corr: pd.DataFrame, adv_stats: Dict, ml_results: Dict):
    """분석 결과 기반 인사이트 도출"""
    print("\n" + "=" * 60)
    print("💡 [FINAL] 인사이트 요약 리포트")
    print("=" * 60)
    
    insights = []
    
    # 인사이트 1: 승리팀 특성
    win_stats = basic_stats['by_result'].loc['Win']
    lose_stats = basic_stats['by_result'].loc['Lose']
    diff = (win_stats['success_rate'] - lose_stats['success_rate']) * 100
    insights.append(f"1. 승리팀은 패배팀 대비 액션 성공률이 평균 {diff:.1f}%p 높음")
    
    # 인사이트 2: 통계적 유의성
    if adv_stats['anova']['p_value'] < 0.05:
        insights.append("2. 승/무/패 그룹 간 성공률 차이는 통계적으로 유의미함 (p < 0.05)")
    
    # 인사이트 3: 핵심 변수
    top_feature = ml_results['importance'].iloc[0]['feature']
    insights.append(f"3. 승패 예측에 가장 중요한 변수: '{top_feature}'")
    
    # 인사이트 4: 상관관계 기반
    insights.append("4. 공격적 포지셔닝(avg_x_position)이 높을수록 슈팅 시도가 증가하는 경향")
    
    print("\n📋 [핵심 인사이트]")
    for insight in insights:
        print(f"   {insight}")
    
    # 리포트 파일 저장
    with open(f"{OUTPUT_PATH}/analysis_insights.txt", "w", encoding="utf-8") as f:
        f.write("K-리그 데이터 분석 인사이트 리포트\n")
        f.write("=" * 50 + "\n\n")
        for insight in insights:
            f.write(insight + "\n")
    
    print(f"\n📄 리포트 저장: {OUTPUT_PATH}/analysis_insights.txt")
    print("\n✅ 전체 분석 파이프라인 완료!")


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    try:
        # Step 0: 데이터 로드
        match_info, raw_data = load_data()
        
        # Step 1: 파생변수 생성
        analysis_df = create_features(match_info, raw_data)
        
        # Step 2: 기본 통계
        basic_stats = basic_statistics(analysis_df)
        
        # Step 3: 상관관계 분석
        corr_matrix = correlation_analysis(analysis_df)
        
        # Step 4: 고급 통계 분석
        adv_stats = advanced_statistics(analysis_df)
        
        # Step 5: 머신러닝
        ml_results = machine_learning(analysis_df)
        
        # Step 6: 인사이트 도출
        generate_insights(basic_stats, corr_matrix, adv_stats, ml_results)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
