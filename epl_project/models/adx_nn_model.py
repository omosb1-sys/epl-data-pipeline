
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os

# --- 1. Data Processing Logic (Inherited from experiment_adx_momentum.py) ---
def calculate_adx_features(df, lookback=5):
    # Sort by team and game_id to maintain time series
    df = df.sort_values(['team', 'game_id'])
    
    # Synthetic OHLC from previous experiment
    df['close'] = df.groupby('team')['points'].cumsum()
    df['high'] = df['close'] + df['goals_for']
    df['low'] = df['close'] - df['goals_against']
    
    features_list = []
    targets_list = []
    
    for team, team_df in df.groupby('team'):
        high = team_df['high']
        low = team_df['low']
        close = team_df['close']
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = pd.Series(0.0, index=team_df.index)
        minus_dm = pd.Series(0.0, index=team_df.index)
        
        plus_mask = (up_move > down_move) & (up_move > 0)
        plus_dm[plus_mask] = up_move[plus_mask]
        minus_mask = (down_move > up_move) & (down_move > 0)
        minus_dm[minus_mask] = down_move[minus_mask]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(lookback).mean()
        plus_di = 100 * (plus_dm.rolling(lookback).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(lookback).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(lookback).mean()
        
        team_df['ADX'] = adx
        team_df['+DI'] = plus_di
        team_df['-DI'] = minus_di
        
        # Target: Result (Win=2, Draw=1, Loss=0)
        team_df['target'] = team_df['result'].map({'Win': 2, 'Draw': 1, 'Loss': 0})
        
        # Predict NEXT match result based on CURRENT ADX
        team_df['next_target'] = team_df['target'].shift(-1)
        
        # Drop rows with NaN (due to rolling or shift)
        valid_data = team_df.dropna(subset=['ADX', '+DI', '-DI', 'next_target'])
        features_list.append(valid_data[['ADX', '+DI', '-DI']])
        targets_list.append(valid_data['next_target'])
        
    X = pd.concat(features_list).values
    y = pd.concat(targets_list).values
    return X, y

# --- 2. PyTorch Dataset ---
class MatchDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 3. Neural Network Model ---
class ADXForecaster(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16, output_dim=3):
        super(ADXForecaster, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# --- 4. Training Engine ---
def train_model():
    print("📈 Extracting ADX features for Deep Learning...")
    df = pd.read_csv('data/processed/team_match_results.csv')
    X, y = calculate_adx_features(df)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    dataset = MatchDataset(X_scaled, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = ADXForecaster()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("🚀 Starting Training (AI Docs Linkage Pattern)...")
    epochs = 50
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
            
    # RandomForest Training (For Ensemble & SHAP)
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    print("🌲 Training RandomForest for ADX Ensemble...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_scaled, y)
    
    # Save Model, RandomForest & Scaler
    torch.save(model.state_dict(), 'epl_project/models/adx_nn_weights.pth')
    joblib.dump(rf_model, 'epl_project/models/adx_rf.pkl')
    joblib.dump(scaler, 'epl_project/models/adx_scaler.pkl')
    
    print("✅ ADX forecasting assets saved (Torch, RF, Scaler)")
    
    return model, rf_model, scaler


if __name__ == "__main__":
    if not os.path.exists('epl_project/models'):
        os.makedirs('epl_project/models')
    train_model()
