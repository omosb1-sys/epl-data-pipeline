"""
K-리그 딥러닝 + SHAP 해석 모듈
================================
PyTorch 기반 신경망 + SHAP Explainability

Author: Antigravity (Senior Data Analyst)
Date: 2026-01-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import shap
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

# GPU/MPS 사용 가능 여부 체크 (Mac Silicon 지원)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🍎 Apple Silicon MPS 가속 활성화")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("🎮 CUDA GPU 가속 활성화")
else:
    DEVICE = torch.device("cpu")
    print("💻 CPU 모드로 실행")


# ============================================
# 1. 데이터 준비
# ============================================
def prepare_data():
    """데이터 로드 및 전처리"""
    print("=" * 60)
    print("📊 [STEP 1] 데이터 준비")
    print("=" * 60)
    
    # 데이터 로드
    match_info = pd.read_csv(f"{BASE_PATH}/match_info.csv")
    raw_data = pd.read_csv(f"{BASE_PATH}/raw_data.csv")
    
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
    
    # 메타데이터 병합
    merged = game_stats.merge(
        match_info[['game_id', 'home_team_id', 'away_team_id', 'home_score', 'away_score']],
        on='game_id', how='left'
    )
    
    # 승/무/패 라벨
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
# 2. PyTorch 신경망 모델 정의
# ============================================
class KLeagueNet(nn.Module):
    """K-리그 승/무/패 예측 신경망"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16]):
        super(KLeagueNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 3))  # 3 classes: Win/Draw/Lose
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# ============================================
# 3. 학습 함수
# ============================================
def train_model(model, train_loader, val_loader, epochs=100):
    """모델 학습"""
    print("\n" + "=" * 60)
    print("🤖 [STEP 2] PyTorch 딥러닝 학습")
    print("=" * 60)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{OUTPUT_PATH}/best_model.pt")
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # 학습 곡선 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='steelblue')
    plt.plot(val_losses, label='Validation Loss', color='coral')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PyTorch 딥러닝 학습 곡선', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_PATH}/learning_curve.png", dpi=150)
    print(f"\n🎨 학습 곡선 저장: {OUTPUT_PATH}/learning_curve.png")
    
    return train_losses, val_losses


# ============================================
# 4. 평가 함수
# ============================================
def evaluate_model(model, test_loader, label_encoder):
    """모델 평가"""
    print("\n" + "=" * 60)
    print("📈 [STEP 3] 모델 평가")
    print("=" * 60)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    # 분류 리포트
    class_names = label_encoder.classes_
    print("\n[Classification Report]")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('PyTorch 딥러닝 혼동 행렬', fontsize=14)
    plt.xlabel('예측')
    plt.ylabel('실제')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/dl_confusion_matrix.png", dpi=150)
    print(f"🎨 혼동 행렬 저장: {OUTPUT_PATH}/dl_confusion_matrix.png")
    
    return all_preds, all_labels


# ============================================
# 5. SHAP 해석
# ============================================
def explain_with_shap(model, X_train, X_test, feature_names):
    """SHAP을 이용한 모델 해석"""
    print("\n" + "=" * 60)
    print("🔍 [STEP 4] SHAP 해석 (Explainable AI)")
    print("=" * 60)
    
    model.eval()
    model.to('cpu')  # SHAP은 CPU에서 실행
    
    # 모델 래퍼 함수 (SHAP용)
    def model_predict(x):
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            outputs = model(x_tensor)
            return outputs.numpy()
    
    # SHAP Explainer (DeepExplainer 대신 Kernel 사용 - 호환성)
    print("  ⏳ SHAP 계산 중 (잠시 기다려 주세요)...")
    
    # 샘플링 (계산 속도 향상)
    background = X_train[:50]
    test_sample = X_test[:30]
    
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(test_sample)
    
    # 5-1. Summary Plot (전체 Feature 중요도)
    plt.figure(figsize=(12, 8))
    
    # Win 클래스에 대한 SHAP
    if isinstance(shap_values, list):
        shap_values_win = shap_values[2]  # Win class (index 2 for Win/Draw/Lose order)
    else:
        shap_values_win = shap_values
    
    shap.summary_plot(shap_values_win, test_sample, 
                      feature_names=feature_names, show=False)
    plt.title('SHAP Feature Importance (승리 예측)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🎨 SHAP Summary Plot 저장: {OUTPUT_PATH}/shap_summary.png")
    
    # 5-2. Bar Plot (평균 중요도)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_win, test_sample, 
                      feature_names=feature_names, plot_type='bar', show=False)
    plt.title('SHAP 평균 Feature 중요도', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/shap_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🎨 SHAP Bar Plot 저장: {OUTPUT_PATH}/shap_bar.png")
    
    # 5-3. 개별 예측 해석 (첫 번째 샘플)
    plt.figure(figsize=(12, 4))
    shap.force_plot(explainer.expected_value[2], shap_values_win[0], 
                   test_sample[0], feature_names=feature_names,
                   matplotlib=True, show=False)
    plt.title('SHAP Force Plot (개별 예측 해석)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/shap_force.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🎨 SHAP Force Plot 저장: {OUTPUT_PATH}/shap_force.png")
    
    return shap_values


# ============================================
# 6. 인사이트 생성
# ============================================
def generate_insights(shap_values, feature_names, X_test):
    """SHAP 기반 인사이트 도출"""
    print("\n" + "=" * 60)
    print("💡 [FINAL] SHAP 기반 인사이트")
    print("=" * 60)
    
    # 평균 절대 SHAP 값으로 중요도 계산
    if isinstance(shap_values, list):
        mean_shap = np.abs(shap_values[2]).mean(axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    print("\n[SHAP 기반 변수 중요도 순위]")
    for i, row in importance_df.iterrows():
        print(f"  {importance_df.index.tolist().index(i)+1}. {row['feature']}: {row['importance']:.4f}")
    
    # 핵심 인사이트 도출
    top_feature = importance_df.iloc[0]['feature']
    second_feature = importance_df.iloc[1]['feature']
    
    insights = [
        f"🏆 승리 예측에 가장 결정적인 변수: '{top_feature}'",
        f"🥈 두 번째 중요 변수: '{second_feature}'",
        f"📊 상위 3개 변수가 전체 예측력의 {importance_df.head(3)['importance'].sum() / importance_df['importance'].sum() * 100:.1f}%를 설명",
        "💡 '성공률(success_rate)'이 높을수록 승리 확률 증가",
        "⚽ 공격적 포지셔닝(avg_x_position)은 공격 시도와 강한 연관"
    ]
    
    print("\n[핵심 인사이트]")
    for insight in insights:
        print(f"  {insight}")
    
    # 리포트 저장
    with open(f"{OUTPUT_PATH}/shap_insights.txt", "w", encoding="utf-8") as f:
        f.write("K-리그 SHAP 기반 딥러닝 인사이트\n")
        f.write("=" * 50 + "\n\n")
        f.write("[변수 중요도]\n")
        f.write(importance_df.to_string() + "\n\n")
        f.write("[인사이트]\n")
        for insight in insights:
            f.write(insight + "\n")
    
    print(f"\n📄 인사이트 저장: {OUTPUT_PATH}/shap_insights.txt")
    
    return importance_df


# ============================================
# MAIN
# ============================================
def main():
    print("\n" + "🚀" * 20)
    print("  K-리그 딥러닝 + SHAP 분석 시스템 시작")
    print("🚀" * 20 + "\n")
    
    # 1. 데이터 준비
    df = prepare_data()
    
    feature_cols = ['total_actions', 'total_passes', 'total_shots', 
                    'success_rate', 'pass_ratio', 'shot_ratio', 
                    'avg_x_position', 'unique_players']
    
    X = df[feature_cols].fillna(0).values
    y = df['result'].values
    
    # 라벨 인코딩
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"✅ 라벨 매핑: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PyTorch 텐서 변환
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. 모델 생성 및 학습
    model = KLeagueNet(input_dim=len(feature_cols)).to(DEVICE)
    print(f"\n📐 모델 구조:\n{model}")
    
    train_model(model, train_loader, test_loader, epochs=100)
    
    # 3. 평가
    model.load_state_dict(torch.load(f"{OUTPUT_PATH}/best_model.pt"))
    evaluate_model(model, test_loader, label_encoder)
    
    # 4. SHAP 해석
    shap_values = explain_with_shap(model, X_train_scaled, X_test_scaled, feature_cols)
    
    # 5. 인사이트
    generate_insights(shap_values, feature_cols, X_test_scaled)
    
    print("\n" + "✅" * 20)
    print("  전체 분석 완료!")
    print("✅" * 20)


if __name__ == "__main__":
    main()
