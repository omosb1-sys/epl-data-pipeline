
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# --- Model Definition (Must match adx_nn_model.py) ---
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

def calculate_adx_single(team_name, lookback=5):
    """
    Analyzes the current ADX trend for a specific team.
    """
    df = pd.read_csv('data/processed/team_match_results.csv')
    team_df = df[df['team'] == team_name].copy()
    team_df = team_df.sort_values('game_id')
    
    # Synthetic OHLC
    team_df['close'] = team_df['points'].cumsum()
    team_df['high'] = team_df['close'] + team_df['goals_for']
    team_df['low'] = team_df['close'] - team_df['goals_against']
    
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
    
    # Get last valid features
    last_features = np.array([[adx.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]]])
    return last_features

def predict_match(team_name):
    print(f"🔮 Analyzing {team_name}'s Momentum for Next Match...")
    
    # 1. Load Model
    model = ADXForecaster()
    weights_path = 'epl_project/models/adx_nn_weights.pth'
    if not os.path.exists(weights_path):
        return "Model weights not found. Please train the model first."
        
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    # 2. Get Real-time Features (Mocking from CSV for now)
    try:
        features = calculate_adx_single(team_name)
        
        # Simple Scaling (Manual mock of scaler state for demo)
        # In production, we would load the saved scaler
        features_tensor = torch.FloatTensor(features)
        
        # 3. Predict
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            
        res_map = {2: "Win", 1: "Draw", 0: "Loss"}
        prob_win = probabilities[0][2].item() * 100
        prob_draw = probabilities[0][1].item() * 100
        prob_loss = probabilities[0][0].item() * 100
        
        report = f"""
### 📈 {team_name} Deep Learning Prediction Report

**[Input Tokens: Current Momentum (ADX)]**
- ADX (Strength): {features[0][0]:.2f}
- +DI (Attack): {features[0][1]:.2f}
- -DI (Defense): {features[0][2]:.2f}

**[Neural Network Forecasting Result]**
- **Predicted Outcome**: {res_map[prediction]}
- **Confidence Probabilities**:
    - Win: {prob_win:.1f}%
    - Draw: {prob_draw:.1f}%
    - Loss: {prob_loss:.1f}%

**[Senior Analyst Commentary]**
{team_name}의 현재 ADX 수치는 {'강한 추세' if features[0][0] > 25 else '보통의 흐름'}를 보이고 있습니다. 
딥러닝 분석 결과, {'승리 확률이 높게 점쳐지며 기세를 이어갈 것' if prediction == 2 else '무승부나 패배의 위험이 감지되어 전술적 재정비가 필요' }으로 보입니다.

---
**#Knowledge_Link**
*   **Tactical Asset**: `research/readings/premier_league_tactical_trends_2024_25.md`
*   **Model Asset**: `epl_project/models/adx_nn_model.py` (PyTorch MLP)
*   **Prediction Time**: 2026-01-17 20:10
"""
        print(report)
        return report

    except Exception as e:
        print(f"Prediction Error: {e}")
        return f"Error analyzing {team_name}"

if __name__ == "__main__":
    import sys
    team = sys.argv[1] if len(sys.argv) > 1 else "Ulsan HD FC"
    predict_match(team)
