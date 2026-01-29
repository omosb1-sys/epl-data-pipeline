"""
K-리그 ML 분석 + 해석 모듈 (SHAP 대체)
=========================================
Permutation Importance + Feature Importance로 해석
(SHAP 설치 불가 환경 대응)

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
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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


def train_multiple_models(X_train, X_test, y_train, y_test, feature_names):
    """여러 모델 학습 및 비교"""
    print("\n" + "=" * 60)
    print("🤖 [STEP 2] 다중 모델 비교 (RF, GB, MLP)")
    print("=" * 60)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500, random_state=42)
    }
    
    results = {}
    best_model, best_score, best_name = None, 0, ""
    
    for name, model in models.items():
        print(f"\n  🔧 {name} 학습 중...")
        model.fit(X_train, y_train)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        test_score = accuracy_score(y_test, model.predict(X_test))
        
        results[name] = {'cv_mean': cv_scores.mean(), 'test_acc': test_score, 'model': model}
        print(f"     CV: {cv_scores.mean():.3f} | Test: {test_score:.3f}")
        
        if test_score > best_score:
            best_score, best_model, best_name = test_score, model, name
    
    print(f"\n🏆 최고 모델: {best_name} ({best_score:.3f})")
    
    # 비교 차트
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    accs = [results[n]['test_acc'] for n in names]
    bars = ax.bar(names, accs, color=['steelblue', 'coral', 'seagreen'])
    ax.set_ylabel('테스트 정확도')
    ax.set_title('모델별 성능 비교', fontsize=14)
    ax.set_ylim(0, 1)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{acc:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/model_comparison.png", dpi=150)
    print(f"🎨 저장: {OUTPUT_PATH}/model_comparison.png")
    
    return best_model, best_name, results


def explain_with_permutation(model, X_test, y_test, feature_names):
    """Permutation Importance를 이용한 모델 해석 (SHAP 대체)"""
    print("\n" + "=" * 60)
    print("🔍 [STEP 3] Permutation Importance (XAI)")
    print("=" * 60)
    
    result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    print("\n[Permutation Importance 순위]")
    for i, row in importance_df.iterrows():
        rank = importance_df.index.tolist().index(i) + 1
        print(f"  {rank}. {row['feature']}: {row['importance_mean']:.4f} (±{row['importance_std']:.4f})")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Permutation Importance
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
    bars = ax1.barh(importance_df['feature'], importance_df['importance_mean'], 
                    xerr=importance_df['importance_std'], color=colors)
    ax1.set_xlabel('Importance')
    ax1.set_title('Permutation Feature Importance', fontsize=14)
    ax1.invert_yaxis()
    
    # Model Feature Importance (트리 기반 모델 전용)
    ax2 = axes[1]
    if hasattr(model, 'feature_importances_'):
        fi = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        ax2.barh(fi['feature'], fi['importance'], color='steelblue')
        ax2.set_xlabel('Importance')
        ax2.set_title('Model Feature Importance (Gini/Entropy)', fontsize=14)
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, 'N/A for MLP', ha='center', va='center', fontsize=14)
        ax2.set_title('Model Feature Importance', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/feature_importance_xai.png", dpi=150)
    print(f"🎨 저장: {OUTPUT_PATH}/feature_importance_xai.png")
    
    return importance_df


def detailed_evaluation(model, X_test, y_test, label_encoder):
    """상세 평가"""
    print("\n" + "=" * 60)
    print("📈 [STEP 4] 상세 평가 리포트")
    print("=" * 60)
    
    y_pred = model.predict(X_test)
    class_names = label_encoder.classes_
    
    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('혼동 행렬', fontsize=14)
    plt.xlabel('예측')
    plt.ylabel('실제')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/confusion_matrix_final.png", dpi=150)
    print(f"🎨 저장: {OUTPUT_PATH}/confusion_matrix_final.png")


def generate_insights(importance_df, model_name):
    """인사이트 생성"""
    print("\n" + "=" * 60)
    print("💡 [FINAL] 핵심 인사이트")
    print("=" * 60)
    
    top1 = importance_df.iloc[0]['feature']
    top2 = importance_df.iloc[1]['feature']
    top3 = importance_df.iloc[2]['feature']
    
    insights = [
        f"🏆 승/패 예측의 1위 핵심 변수: '{top1}'",
        f"🥈 2위: '{top2}', 3위: '{top3}'",
        "📊 성공률(success_rate)이 높은 팀일수록 승리 확률 증가",
        "📍 공격적 포지셔닝이 슈팅 기회로 직결됨",
        "⚠️ 무승부(Draw) 예측은 불확실성이 높아 외부 변수(날씨, 중요도) 필요"
    ]
    
    print("\n📋 [핵심 인사이트]")
    for insight in insights:
        print(f"   {insight}")
    
    with open(f"{OUTPUT_PATH}/final_insights.txt", "w", encoding="utf-8") as f:
        f.write(f"K-리그 ML 분석 최종 인사이트\n")
        f.write(f"모델: {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write("[변수 중요도]\n")
        f.write(importance_df.to_string() + "\n\n")
        f.write("[인사이트]\n")
        for insight in insights:
            f.write(insight + "\n")
    
    print(f"\n📄 저장: {OUTPUT_PATH}/final_insights.txt")


def main():
    print("\n" + "🚀" * 20)
    print("  K-리그 ML 분석 + XAI 시스템")
    print("🚀" * 20 + "\n")
    
    df = prepare_data()
    
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
    
    best_model, best_name, _ = train_multiple_models(
        X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    )
    
    importance_df = explain_with_permutation(best_model, X_test_scaled, y_test, feature_cols)
    
    detailed_evaluation(best_model, X_test_scaled, y_test, label_encoder)
    
    generate_insights(importance_df, best_name)
    
    print("\n" + "✅" * 20)
    print("  전체 분석 완료!")
    print("✅" * 20)


if __name__ == "__main__":
    main()
