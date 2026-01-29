
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

class MovingAvg(nn.Module):
    """시계열 데이터의 추세를 추출하기 위한 이동 평균 필터 (Moving Average)"""
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: [batch, seq_len, channels]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class DLinearForecaster(nn.Module):
    """
    DLinear: 시계열을 Trend와 Seasonal로 분리하여 학습하는 고성능 선형 모델
    'Simple is the Best' 원칙을 따르며, 딥러닝 분석가들이 가장 선호하는 베이스라인입니다.
    """
    def __init__(self, seq_len, pred_len, enc_in):
        super(DLinearForecaster, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in

        # Decomposition (추세와 계절성 분해)
        self.decompsition = MovingAvg(kernel_size=5, stride=1) # 최근 5경기 기준
        
        # 선형 레이어 (각 채널 독립적으로 학습 - Channel Independence)
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        trend_init = self.decompsition(x)
        seasonal_init = x - trend_init
        
        # Seasonal 경로
        seasonal_init = seasonal_init.permute(0, 2, 1) # [B, C, L]
        seasonal_output = self.linear_seasonal(seasonal_init)
        seasonal_output = seasonal_output.permute(0, 2, 1) # [B, Pred_L, C]
        
        # Trend 경로
        trend_init = trend_init.permute(0, 2, 1)
        trend_output = self.linear_trend(trend_init)
        trend_output = trend_output.permute(0, 2, 1)
        
        # 최종 결합
        return seasonal_output + trend_output

def prepare_patch_data(team_history: list, patch_size: int = 5):
    """
    데이터 분석가용 도구: 개별 경기 데이터를 패치(최근 N경기) 단위로 변환
    """
    if len(team_history) < patch_size:
        # 데이터가 부족하면 패딩 처리
        padding = [team_history[0]] * (patch_size - len(team_history))
        team_history = padding + team_history
    
    return torch.tensor(team_history[-patch_size:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

if __name__ == "__main__":
    # 데이터 분석가용 프로토타입 테스트
    print("🚀 [DLinear Forecaster] 프로토타입 시뮬레이션 시작...")
    
    # 가상의 최근 5경기 득점 데이터 (Patch)
    # [1, 2, 0, 3, 1] -> 흐름 분석 시작
    mock_history = [1.0, 2.0, 0.0, 3.0, 1.0]
    input_tensor = prepare_patch_data(mock_history)
    
    # 모델 초기화 (입력 5경기 -> 향후 1경기 예측, 채널 1개)
    model = DLinearForecaster(seq_len=5, pred_len=1, enc_in=1)
    
    prediction = model(input_tensor)
    print(f"✅ 입력 패치: {mock_history}")
    print(f"🔮 예측된 다음 경기 득점: {prediction.item():.2f}")
    print("""
    💡 분석가 노트:
    DLinear 모델은 '최근 경기력의 관성(Trend)'과 '불규칙한 변동(Seasonal)'을 
    수학적으로 분리하여 읽어냅니다. 이는 단순 평균보다 팀의 흐름을 정확히 포착합니다.
    """)
