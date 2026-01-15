import os

# [Stability Patch] Mac M시리즈 라이브러리 충돌 및 메모리 에러 완전 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1' # 멀티스레딩 충돌 방지 (안정성 최우선)

def get_ensemble_engine():
    # 무거운 라이브러리는 분석 직전에만 로드 (Lazy Loading)
    try:
        import torch
        import torch.nn as nn
        import joblib
        import numpy as np
        # LightGBM 대신 더 가볍고 표준적인 RandomForest 사용 (Mac 안정성 끝판왕)
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        return None, None, None

    class EPLDeepNet(nn.Module):
        def __init__(self, input_size):
            super(EPLDeepNet, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 64), # 레이어 단순화로 부하 감소
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            self.eval()
            with torch.no_grad():
                return self.net(x)

    BASE_DIR = os.path.dirname(__file__)
    torch_path = os.path.join(BASE_DIR, "models/epl_pytorch.pth")
    rf_path = os.path.join(BASE_DIR, "models/epl_rf.pkl")
    scaler_path = os.path.join(BASE_DIR, "models/scaler.pkl")
    
    try:
        if all(os.path.exists(p) for p in [torch_path, rf_path, scaler_path]):
            model_torch = EPLDeepNet(input_size=4)
            model_torch.load_state_dict(torch.load(torch_path, map_location=torch.device('cpu')))
            model_torch.eval()
            
            model_rf = joblib.load(rf_path)
            scaler = joblib.load(scaler_path)
            return model_torch, model_rf, scaler
    except:
        pass
    
    return None, None, None
