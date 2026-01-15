import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from datetime import datetime

# =============================================================
# 🏆 Next-Level Ensemble Engine (PyTorch + XGBoost + ELO)
# =============================================================

# 1. ELO Rating System Implementation
def calculate_elo(rating_a, rating_b, actual_score, k_factor=32):
    """
    ELO 점수 계산기 (실력 기반 지수)
    actual_score: 1(Home Win), 0.5(Draw), 0(Away Win)
    """
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    new_rating_a = rating_a + k_factor * (actual_score - expected_a)
    return round(new_rating_a, 2)

# 2. Deep Learning Model (PyTorch)
class EPLDeepNet(nn.Module):
    def __init__(self, input_size):
        super(EPLDeepNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def train_next_level_ensemble():
    print("🚀 [Expert Engine] Ensemble & ELO 시스템 구축 및 학습 시작...")
    
    BASE_DIR = os.path.dirname(__file__)
    data_path = os.path.join(BASE_DIR, "data/advanced/team_advanced_stats.json")
    
    # [A] 데이터 준비 (ELO 지수 가상 생성 및 피처링)
    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            df_raw = pd.DataFrame(json.load(f))
        
        # 가공 피처 생성
        df_raw['target'] = (df_raw.get('goals_scored', 0) > df_raw.get('goals_conceded', 0)).astype(float)
        df_raw['form_score'] = df_raw.get('form', 'DDDDD').apply(lambda x: sum([3 if c=='W' else 1 if c=='D' else 0 for c in x[-5:]])/15.0 if isinstance(x, str) else 0.5)
        if 'elo' not in df_raw.columns: df_raw['elo'] = 1500

        # XGBoost 학습을 위해 최소한 0과 1 클래스가 모두 있는지 확인
        if len(df_raw['target'].unique()) < 2:
            print("⚠️ 실제 데이터의 클래스가 부족하여 합성 데이터를 혼합합니다.")
            synth_df = pd.DataFrame([
                {"goals_scored": 60, "goals_conceded": 10, "elo": 1800, "form_score": 0.9, "target": 1},
                {"goals_scored": 10, "goals_conceded": 60, "elo": 1200, "form_score": 0.1, "target": 0}
            ])
            df = pd.concat([df_raw, synth_df], ignore_index=True)
        else:
            df = df_raw
    else:
        # 완전 합성 데이터
        df = pd.DataFrame([
            {"goals_scored": 70, "goals_conceded": 5, "elo": 1900, "form_score": 0.95, "target": 1},
            {"goals_scored": 5, "goals_conceded": 70, "elo": 1100, "form_score": 0.05, "target": 0},
            {"goals_scored": 50, "goals_conceded": 20, "elo": 1750, "form_score": 0.8, "target": 1},
            {"goals_scored": 20, "goals_conceded": 50, "elo": 1350, "form_score": 0.2, "target": 0},
            {"goals_scored": 35, "goals_conceded": 35, "elo": 1550, "form_score": 0.5, "target": 0.5}
        ] * 40)

    features = ['goals_scored', 'goals_conceded', 'elo', 'form_score']
    # 누락된 컬럼 보안
    for feat in features:
        if feat not in df.columns: df[feat] = 50 if "goals" in feat else 1500 if "elo" in feat else 0.5
    
    X = df[features].values.astype(np.float32)
    y = df['target'].values.astype(np.float32)

    # 1. 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. PyTorch 학습
    print("🤖 PyTorch Deep Net 학습 중...")
    model_torch = EPLDeepNet(input_size=len(features))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_torch.parameters(), lr=0.005)

    inputs = torch.from_numpy(X_scaled)
    labels = torch.from_numpy(y.reshape(-1, 1))

    for epoch in range(150):
        optimizer.zero_grad()
        outputs = model_torch(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 3. XGBoost 학습 (강력한 수학적 분류기)
    print("🌪️ XGBoost Classifier 학습 중...")
    model_xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5)
    # XGBoost는 이진 분류를 위해 target을 0 or 1로 변환
    y_binary = (y > 0.5).astype(int)
    model_xgb.fit(X_scaled, y_binary)

    # [B] 저장
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model_torch.state_dict(), os.path.join(model_dir, "epl_pytorch.pth"))
    joblib.dump(model_xgb, os.path.join(model_dir, "epl_xgb.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    
    print(f"✨ Ensemble 모델 세트 저장 완료: {model_dir}")

if __name__ == "__main__":
    train_next_level_ensemble()
