"""
📊 K리그 데이터 분석 파이프라인 - Task 2: 심층 EDA 및 통계 검증
========================================================================
이 스크립트는 정제된 데이터를 바탕으로 변수 간 상관관계를 분석하고,
주요 지표들이 승리에 미치는 영향을 통계적으로 검증(ANOVA)합니다.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

def task2_eda_and_stats(data_path):
    """
    고급 시각화 및 통계 분석을 수행하는 메인 함수
    """
    print("--- Task 2: Advanced EDA & Statistical Validation ---")
    
    # 1. 데이터 로드
    df = pd.read_csv(data_path)
    
    # 시각화 스타일 설정 (한글 폰트 포함)
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 2. 상관관계 분석 (Correlation Analysis)
    print("📊 주요 지표 간 상관관계 히트맵 생성 중...")
    cols_to_corr = [
        'total_passes', 'pass_success_rate', 'total_shots', 
        'tackles', 'interceptions', 'attack_zone_actions', 
        'goals', 'is_win', 'shot_efficiency', 'defensive_pressure',
        'rolling_win_rate', 'rolling_pass_rate'
    ]
    plt.figure(figsize=(12, 10))
    corr = df[cols_to_corr].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('K-리그 주요 지표 상관관계 히트맵', fontsize=16)
    
    corr_img = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/advanced_correlation_heatmap.png"
    plt.savefig(corr_img, dpi=300)
    plt.close()
    print(f"✓ 히트맵 저장 완료: {corr_img}")

    # 3. 통계적 유의성 검정 (ANOVA)
    # 질문: '패스 성공률'은 승리 팀과 비승리 팀 간에 통계적으로 유의미한 차이가 있는가?
    print("🧪 통계적 유의성 검정(ANOVA) 수행 중...")
    win_pass = df[df['is_win'] == 1]['pass_success_rate']
    no_win_pass = df[df['is_win'] == 0]['pass_success_rate']
    
    # ANOVA 테스트 실행
    f_val, p_val = stats.f_oneway(win_pass, no_win_pass)
    print(f"✓ ANOVA 결과 (패스 성공률 vs 승리): F={f_val:.2f}, p={p_val:.4f}")

    # 시각적 검증을 위한 박스플롯(Boxplot) 생성
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='is_win', y='pass_success_rate', data=df, palette='Set2')
    plt.title(f'승리 여부에 따른 패스 성공률 분포 (p-value: {p_val:.4f})', fontsize=14)
    plt.xlabel('승리/무승부 여부 (0: 패배, 1: 승리/무승부)')
    plt.ylabel('패스 성공률 (%)')
    
    stats_img = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/statistical_validation.png"
    plt.savefig(stats_img, dpi=300)
    plt.close()
    print(f"✓ 통계 검증 차트 저장 완료: {stats_img}")

    return {"f_val": f_val, "p_val": p_val}

if __name__ == "__main__":
    DATA_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/processed/processed_ml_data.csv"
    if os.path.exists(DATA_PATH):
        task2_eda_and_stats(DATA_PATH)
    else:
        print("정제된 데이터 파일을 찾을 수 없습니다. Task 1을 먼저 실행하세요.")
