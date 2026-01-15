import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import os
import shap
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# ==========================================
# 🧠 EPL Deep Learning Predictor (MLP Version)
# ==========================================

class EPLPredictorNet(nn.Module):
    def __init__(self, input_size):
        super(EPLPredictorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # 0~1 사이 승률 반환
        )
    
    def forward(self, x):
        return self.net(x)

def train_and_save_model():
    print("🚀 [Training Engine] Deep Learning 모델 학습 시작...")
    
    BASE_DIR = os.path.dirname(__file__)
    data_path = os.path.join(BASE_DIR, "data/advanced/team_advanced_stats.json")
    
    # 1. 데이터 로드 및 전처리
    if not os.path.exists(data_path):
        # 만약 데이터 없으면 검증용 더미 데이터로 학습 (추후 실제 데이터로 덮어쓰기)
        df = pd.DataFrame([
            {"goals_scored": 50, "goals_conceded": 15, "power_index": 90, "form_score": 0.9, "target": 1},
            {"goals_scored": 45, "goals_conceded": 20, "power_index": 88, "form_score": 0.8, "target": 1},
            {"goals_scored": 20, "goals_conceded": 40, "power_index": 60, "form_score": 0.3, "target": 0},
            {"goals_scored": 30, "goals_conceded": 35, "power_index": 75, "form_score": 0.5, "target": 0.5},
            {"goals_scored": 55, "goals_conceded": 10, "power_index": 95, "form_score": 1.0, "target": 1},
        ] * 20) # 데이터 증강
    else:
        with open(data_path, 'r') as f:
            df = pd.DataFrame(json.load(f))
            # 가공 피처 생성
            df['target'] = (df['goals_scored'] > df['goals_conceded']).astype(float)
            df['form_score'] = df['form'].apply(lambda x: sum([3 if c=='W' else 1 if c=='D' else 0 for c in x[-5:]])/15.0 if x else 0.5)

    features = ['goals_scored', 'goals_conceded', 'power_index', 'form_score']
    X = df[features].values.astype(np.float32)
    y = df['target'].values.reshape(-1, 1).astype(np.float32)

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 모델 생성 및 학습
    model = EPLPredictorNet(input_size=len(features))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        inputs = torch.from_numpy(X_scaled)
        labels = torch.from_numpy(y)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 모델 및 스케일러 저장
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "epl_model.pth"))
    
    import joblib
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    
    print(f"✅ Deep Learning 모델 저장 완료: {model_dir}")

if __name__ == "__main__":
    train_and_save_model()
