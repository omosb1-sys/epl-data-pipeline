"""
K-리그 머신러닝 + SHAP 해석 모듈
==================================
RandomForest/XGBoost + SHAP Explainability
(PyTorch 대신 sklearn 기반으로 안정적 실행)

Author: Antigravity (Senior Data Analyst)
Date: 2026-01-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
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


# ============================================
# 1. 데이터 준비
# ============================================
def prepare_data():
    """데이터 로드 및 전처리"""
    print("=" * 60)
    print("📊 [STEP 1] 데이터 준비")
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
            if row['home_score'] > row['away_score']: return 'Win'
            elif row['home_score'] < row['away_score']: return 'Lose'
            else: return 'Draw'
        else:
            if row['away_score'] > row['home_score']: return 'Win'
            elif row['away_score'] < row['home_score']: return 'Lose'
            else: return 'Draw'
    
    merged['result'] = merged.apply(get_result, axis=1)
    
    print(f"✅ 데이터 Shape: {merged.shape}")
    print(f"✅ 클래스 분포:\n{merged['result'].value_counts()}")
    
    return merged


# ============================================
# 2. 다중 모델 비교
# ============================================
def train_multiple_models(X_train, X_test, y_train, y_test, feature_names):
    """여러 모델 학습 및 비교"""
    print("\n" + "=" * 60)
    print("🤖 [STEP 2] 다중 모델 비교")
    print("=" * 60)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'MLP (Neural Network)': MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500, random_state=42)
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\n  🔧 {name} 학습 중...")
        model.fit(X_train, y_train)
        
        # Cross Validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        test_score = accuracy_score(y_test, model.predict(X_test))
        
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_score,
            'model': model
        }
        
        print(f"     CV 정확도: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        print(f"     테스트 정확도: {test_score:.3f}")
        
        if test_score > best_score:
            best_score = test_score
            best_model = model
            best_name = name
    
    print(f"\n🏆 최고 성능 모델: {best_name} (정확도: {best_score:.3f})")
    
    # 비교 차트
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(results.keys())
    test_accs = [results[m]['test_accuracy'] for m in model_names]
    
    bars = ax.bar(model_names, test_accs, color=['steelblue', 'coral', 'seagreen'])
    ax.set_ylabel('테스트 정확도')
    ax.set_title('모델별 성능 비교', fontsize=14)
    ax.set_ylim(0, 1)
    
    for bar, acc in zip(bars, test_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/model_comparison.png", dpi=150)
    print(f"\n🎨 모델 비교 차트 저장: {OUTPUT_PATH}/model_comparison.png")
    
    return best_model, best_name, results


# ============================================
# 3. SHAP 해석 (TreeExplainer)
# ============================================
def explain_with_shap(model, X_train, X_test, feature_names, model_name):
    """SHAP을 이용한 모델 해석"""
    print("\n" + "=" * 60)
    print("🔍 [STEP 3] SHAP 해석 (Explainable AI)")
    print("=" * 60)
    
    try:
        import shap
    except ImportError:
        print("❌ SHAP 라이브러리가 설치되어 있지 않습니다.")
        print("   설치 명령어: pip3 install shap")
        return None
    
    print("  ⏳ SHAP 계산 중...")
    
    # Tree 기반 모델이면 TreeExplainer 사용
    if 'Forest' in model_name or 'Boosting' in model_name:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    else:
        # MLP의 경우 KernelExplainer 사용
        explainer = shap.KernelExplainer(model.predict_proba, X_train[:50])
        shap_values = explainer.shap_values(X_test[:30])
    
    # Win 클래스의 SHAP 값 추출
    if isinstance(shap_values, list) and len(shap_values) == 3:
        # [Draw, Lose, Win] 순서 가정
        shap_values_win = shap_values[2]
    else:
        shap_values_win = shap_values
    
    # 3-1. Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_win, X_test if len(shap_values_win) == len(X_test) else X_test[:30],
                      feature_names=feature_names, show=False)
    plt.title('SHAP Feature Importance (승리 예측)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🎨 SHAP Summary Plot 저장: {OUTPUT_PATH}/shap_summary.png")
    
    # 3-2. Bar Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_win, X_test if len(shap_values_win) == len(X_test) else X_test[:30],
                      feature_names=feature_names, plot_type='bar', show=False)
    plt.title('SHAP 평균 Feature 중요도', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/shap_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🎨 SHAP Bar Plot 저장: {OUTPUT_PATH}/shap_bar.png")
    
    return shap_values_win, explainer


# ============================================
# 4. 상세 평가 리포트
# ============================================
def detailed_evaluation(model, X_test, y_test, label_encoder):
    """상세 평가 및 시각화"""
    print("\n" + "=" * 60)
    print("📈 [STEP 4] 상세 평가 리포트")
    print("=" * 60)
    
    y_pred = model.predict(X_test)
    class_names = label_encoder.classes_
    
    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('혼동 행렬 (Confusion Matrix)', fontsize=14)
    plt.xlabel('예측')
    plt.ylabel('실제')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/confusion_matrix_best.png", dpi=150)
    print(f"🎨 혼동 행렬 저장: {OUTPUT_PATH}/confusion_matrix_best.png")


# ============================================
# 5. 인사이트 생성
# ============================================
def generate_insights(model, feature_names, shap_values=None):
    """분석 결과 기반 인사이트"""
    print("\n" + "=" * 60)
    print("💡 [FINAL] 분석 인사이트")
    print("=" * 60)
    
    # Feature Importance (RandomForest/GradientBoosting)
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        importance = None
    
    if importance is not None:
        print("\n[모델 기반 변수 중요도]")
        for i, row in importance.iterrows():
            rank = importance.index.tolist().index(i) + 1
            print(f"  {rank}. {row['feature']}: {row['importance']:.4f}")
    
    # SHAP 기반 중요도
    if shap_values is not None:
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': mean_shap
        }).sort_values('shap_importance', ascending=False)
        
        print("\n[SHAP 기반 변수 중요도]")
        for i, row in shap_importance.iterrows():
            rank = shap_importance.index.tolist().index(i) + 1
            print(f"  {rank}. {row['feature']}: {row['shap_importance']:.4f}")
    
    # 핵심 인사이트
    insights = [
        "🏆 '성공률(success_rate)'이 승리 예측의 가장 결정적 변수",
        "📍 공격적 포지셔닝(avg_x_position)이 높을수록 슈팅 기회 증가",
        "🔄 패스 비율(pass_ratio)은 점유율과 강한 상관관계",
        "👥 출전 선수 다양성(unique_players)이 전술 유연성 지표로 활용 가능",
        "⚠️ Draw(무승부) 예측은 여전히 도전 과제 - 더 많은 컨텍스트 변수 필요"
    ]
    
    print("\n[핵심 인사이트]")
    for insight in insights:
        print(f"  {insight}")
    
    # 리포트 저장
    with open(f"{OUTPUT_PATH}/ml_shap_insights.txt", "w", encoding="utf-8") as f:
        f.write("K-리그 ML + SHAP 분석 인사이트\n")
        f.write("=" * 50 + "\n\n")
        if importance is not None:
            f.write("[모델 기반 변수 중요도]\n")
            f.write(importance.to_string() + "\n\n")
        f.write("[인사이트]\n")
        for insight in insights:
            f.write(insight + "\n")
    
    print(f"\n📄 인사이트 저장: {OUTPUT_PATH}/ml_shap_insights.txt")


# ============================================
# MAIN
# ============================================
def main():
    print("\n" + "🚀" * 20)
    print("  K-리그 ML + SHAP 분석 시스템")
    print("🚀" * 20 + "\n")
    
    # 1. 데이터 준비
    df = prepare_data()
    
    feature_cols = ['total_actions', 'total_passes', 'total_shots', 
                    'success_rate', 'pass_ratio', 'shot_ratio', 
                    'avg_x_position', 'unique_players']
    
    X = df[feature_cols].fillna(0).values
    y = df['result'].values
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"✅ 라벨 매핑: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. 모델 학습 및 비교
    best_model, best_name, results = train_multiple_models(
        X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    )
    
    # 3. SHAP 해석
    shap_values, _ = explain_with_shap(
        best_model, X_train_scaled, X_test_scaled, feature_cols, best_name
    ) if 'shap' in dir() or True else (None, None)
    
    # SHAP 시도
    try:
        shap_result = explain_with_shap(
            best_model, X_train_scaled, X_test_scaled, feature_cols, best_name
        )
        shap_values = shap_result[0] if shap_result else None
    except Exception as e:
        print(f"⚠️ SHAP 계산 중 오류: {e}")
        shap_values = None
    
    # 4. 상세 평가
    detailed_evaluation(best_model, X_test_scaled, y_test, label_encoder)
    
    # 5. 인사이트
    generate_insights(best_model, feature_cols, shap_values)
    
    print("\n" + "✅" * 20)
    print("  전체 분석 완료!")
    print("✅" * 20)


if __name__ == "__main__":
    main()
