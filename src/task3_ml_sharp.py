"""
🤖 K리그 데이터 분석 파이프라인 - Task 3: 머신러닝 모델링 및 ShaRP 해석
========================================================================
이 스크립트는 정제된 데이터를 사용하여 승리 예측 모델을 학습시키고,
ShaRP(Shapley for Rankings) 기술을 사용하여 모델의 판단 근거를 시각적으로 해석합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import sharp
import os

def task3_ml_and_sharp(data_path):
    """
    모델 학습 및 해석을 수행하는 메인 함수
    """
    print("--- Task 3: ML Modeling & ShaRP Interpretation ---")
    
    # 1. 데이터 로드
    df = pd.read_csv(data_path)
    
    # 2. 피처(독립변수) 및 타겟(종속변수) 준비
    # 승리(is_win)를 예측하기 위한 주요 변수들 선택
    features = [
        'pass_success_rate', 'total_shots', 'tackles', 
        'interceptions', 'attack_zone_actions', 'shot_efficiency', 
        'defensive_pressure', 'rolling_win_rate', 'rolling_pass_rate', 'is_home'
    ]
    X = df[features].fillna(0)
    y = df['is_win']
    
    # 학습 데이터와 테스트 데이터 분리 (8:2 비율)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. 모델 학습 (랜덤 포레스트 분류기)
    print("🤖 랜덤 포레스트 분류 모델 학습 중...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. 모델 성능 평가 (ROC-AUC 지표 사용)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"✓ 모델 성능 평가 (ROC-AUC): {auc:.4f}")
    
    # 5. 모델 해석 (ShaRP 및 Fallback 전략)
    # 왜 모델이 특정 팀의 승리를 예측했는지 분석합니다.
    print("🧠 모델 해석(ShaRP) 적용 중...")
    
    interpret_img = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/model_interpretation.png"
    
    try:
        # ShaRP를 위한 점수 함수 정의 (승리 확률 반환)
        def score_func(X_input):
            if isinstance(X_input, np.ndarray):
                X_input = pd.DataFrame(X_input, columns=features)
            return model.predict_proba(X_input)[:, 1]
        
        # ShaRP 객체 초기화
        print("ShaRP (xai-sharp) 라이브러리 사용 중...")
        explainer = sharp.ShaRP(
            qoi="score",
            qoi_func=score_func,
            ref_distribution=X_train.sample(min(50, len(X_train)), random_state=42).values
        )
        
        # 기여도 계산 (테스트 샘플 일부 사용)
        X_sample = X_test.head(20).values
        sharp_values = explainer.all(X_sample)
        
        # 시각화 데이터 프레임 생성
        importance_df = pd.DataFrame({
            'feature': features,
            'contribution': np.abs(sharp_values).mean(axis=0)
        }).sort_values('contribution', ascending=False)
        
        # 막대 그래프 시각화
        plt.figure(figsize=(10, 6))
        sns.barplot(x='contribution', y='feature', data=importance_df, palette='viridis')
        plt.title('ShaRP 모델 해석: K-리그 승리 결정 요인', fontsize=14)
        plt.xlabel('평균 절대 기여도 (Absolute Contribution)')
        plt.savefig(interpret_img, dpi=300)
        plt.close()
        print(f"✓ ShaRP 해석 결과가 저장되었습니다: {interpret_img}")
        
    except Exception as e:
        # ShaRP 실패 시 기본 중요도(Feature Importance)로 대체
        print(f"⚠️ ShaRP 계산 중 오류 발생: {e}")
        print("기본 변수 중요도(Feature Importance) 방식으로 시각화를 대체합니다.")
        
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': features, 
            'contribution': importances
        }).sort_values('contribution', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='contribution', y='feature', data=importance_df, palette='mako')
        plt.title('변수 중요도 (기본 모델 제공)', fontsize=14)
        plt.savefig(interpret_img, dpi=300)
        plt.close()
        print(f"✓ 기본 변수 중요도 차트가 저장되었습니다: {interpret_img}")

    return auc

if __name__ == "__main__":
    DATA_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/processed/processed_ml_data.csv"
    if os.path.exists(DATA_PATH):
        task3_ml_and_sharp(DATA_PATH)
    else:
        print("데이터 파일을 찾을 수 없습니다.")
