import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class EPLDeepNet(nn.Module):
    def __init__(self, input_size):
        super(EPLDeepNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def train_stable_engine():
    print("🚀 [Expert Engine] Super-Stable 시스템 구축 및 학습 시작...")
    BASE_DIR = os.path.dirname(__file__)
    
    # 합성 데이터 생성
    df = pd.DataFrame([
        {"goals_scored": 70, "goals_conceded": 5, "elo": 1900, "form_score": 0.95, "target": 1},
        {"goals_scored": 5, "goals_conceded": 70, "elo": 1100, "form_score": 0.05, "target": 0},
        {"goals_scored": 50, "goals_conceded": 20, "elo": 1750, "form_score": 0.8, "target": 1},
        {"goals_scored": 20, "goals_conceded": 50, "elo": 1350, "form_score": 0.2, "target": 0},
        {"goals_scored": 35, "goals_conceded": 35, "elo": 1550, "form_score": 0.5, "target": 0.5}
    ] * 50)

    features = ['goals_scored', 'goals_conceded', 'elo', 'form_score']
    X = df[features].values.astype(np.float32)
    y = df['target'].values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. PyTorch
    model_torch = EPLDeepNet(input_size=len(features))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_torch.parameters(), lr=0.01)
    inputs = torch.from_numpy(X_scaled)
    labels = torch.from_numpy(y.reshape(-1, 1))

    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model_torch(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 2. RandomForest (안정성 끝판왕)
    print("🌲 RandomForest Classifier 학습 중...")
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_rf.fit(X_scaled, (y > 0.5).astype(int))

    # 저장
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model_torch.state_dict(), os.path.join(model_dir, "epl_pytorch.pth"))
    joblib.dump(model_rf, os.path.join(model_dir, "epl_rf.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print(f"✨ 초안정화 모델 세트 저장 완료")

if __name__ == "__main__":
    train_stable_engine()
