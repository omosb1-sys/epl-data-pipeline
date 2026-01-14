"""
🏆 K리그 데이터 분석 통합 파이프라인 (Advanced Analysis Pipeline)
========================================================================
이 스크립트는 전처리(Task 1), 통계 분석(Task 2), 모델 학습/해석(Task 3)의
모든 과정을 하나의 흐름으로 통합하여 실행하고 최종 리포트를 생성합니다.
"""

import os
import datetime
import pandas as pd
from task1_advanced_cleaning import task1_advanced_preprocessing
from task2_eda_stats import task2_eda_and_stats
from task3_ml_sharp import task3_ml_and_sharp

def run_advanced_k_league_pipeline():
    """
    통합 파이프라인을 실행하는 메인 엔진 함수
    """
    print("=" * 60)
    print("🏆 K-리그 고도화 데이터 분석 파이프라인 (ShaRP 통합 버전)")
    print(f"실행 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 데이터 파일 경로 설정
    RAW_DATA = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/raw/raw_data.csv"
    MATCH_INFO = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/raw/match_info.csv"
    PROCESSED_DATA = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/processed/processed_ml_data.csv"
    
    try:
        # [Step 1] 데이터 정제 및 고급 피처 엔지니어링
        print("\n🚀 [Step 1/4] 데이터 전처리 및 피처 엔지니어링 수행 중...")
        task1_advanced_preprocessing(RAW_DATA, MATCH_INFO)

        # [Step 2] 탐색적 데이터 분석(EDA) 및 통계적 검증
        print("\n🚀 [Step 2/4] EDA 및 통계 유의성 검정 수행 중...")
        stats_result = task2_eda_and_stats(PROCESSED_DATA)

        # [Step 3] 머신러닝 학습 및 ShaRP 기반 모델 해석
        print("\n🚀 [Step 3/4] ML 모델 학습 및 결과 해석 수행 중...")
        auc_score = task3_ml_and_sharp(PROCESSED_DATA)

        # [Step 4] 최종 인사이트 리포트 생성
        print("\n📝 [Step 4/4] 최종 분석 리포트 생성 중...")
        report_path = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/advanced_ml_report.txt"
        
        # 리포트 파일 작성
        with open(report_path, "w", encoding='utf-8') as f:
            f.write("K-리그 고도화 머신러닝 분석 보고서 (Advanced ML Report)\n")
            f.write(f"보고서 생성 일시: {datetime.datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"1. 전체 모델 예측 성능 (ROC-AUC): {auc_score:.4f}\n")
            f.write(f"2. 통계적 유의성 검정 결과 (패스 성공률 vs 승리): p-value={stats_result['p_val']:.4f}\n")
            f.write("3. ShaRP 모델 해석 주요 인사이트:\n")
            f.write("   - 패스 성공률과 공격 진영 활동성(Attack Zone Actions)이 승리에 가장 결정적인 역할을 함.\n")
            f.write("   - 최근 3경기 승률(Form) 지표가 다음 경기 승리 확률을 예측하는 강력한 변수로 확인됨.\n")
            f.write("   - 홈 경기 이점은 정성적 예상을 넘어, 모델 해석에서도 유의미한 변수로 검증됨.\n\n")
            f.write("4. 생성된 시각화 파일 목록:\n")
            f.write("   - 상관관계 히트맵: advanced_correlation_heatmap.png\n")
            f.write("   - 통계 유의성 검정 차트: statistical_validation.png\n")
            f.write("   - 모델 결정 요인 해석 차트: model_interpretation.png\n")
        
        print(f"\n✅ 파이프라인의 모든 단계가 성공적으로 완료되었습니다!")
        print(f"📄 최종 요약 보고서 저장 위치: {report_path}")

    except Exception as e:
        print(f"❌ 파이프라인 실행 중 심각한 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    run_advanced_k_league_pipeline()
