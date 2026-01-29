
# 📈 PyTorch Time-Series Forecasting Expert Skill (DLinear & Patching)

이 스킬은 데이터 분석가의 관점에서 **PyTorch를 활용한 고성능 시계열 예측**을 수행하기 위한 최신 방법론과 구현 가이드를 담고 있습니다.

## 1. 🧠 Core Methodology

### 1.1 Patching (데이터 블록화)
- **개념**: 개별 데이터 포인트가 아닌, 일정 기간(예: 최근 5경기)을 하나의 '패치(Patch)'로 묶어 분석합니다.
- **장점**: 노이즈를 억제하고 데이터의 '흐름(Trend)'을 더 명확하게 포착합니다.
- **적용**: `torch.unfold`를 사용하여 슬라이딩 윈도우 방식으로 데이터를 패칭합니다.

### 1.2 DLinear (Decomposition Linear)
- **개념**: 복잡한 트랜스포머 대신, 시계열 데이터를 **'추세(Trend)'**와 **'계절성(Seasonality)'**으로 분리(Decomposition)한 후 각각 선형 회귀를 적용하는 모델입니다.
- **장점**: 연산이 매우 가볍고, 해석력이 극대화됩니다. (Simple is the best)
- **적용**: `Moving Average` 필터링을 통해 시계열을 분해합니다.

### 1.3 Channel Independence (채널 독립성)
- **개념**: 공격 지표(득점)와 수비 지표(실점)를 섞지 않고 각각 독립적인 채널로 처리한 후 나중에 결합합니다.
- **장점**: 다변량 시계열에서 발생할 수 있는 지표 간 간섭과 오염을 방지합니다.

## 2. 💻 Implementation Template (PyTorch)

```python
import torch
import torch.nn as nn

class MovingAvg(nn.Module):
    \"\"\"시계열 분해를 위한 이동 평균 레이어\"\"\"
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class DLinearModel(nn.Module):
    \"\"\"데이터 분석가를 위한 DLinear 구현체\"\"\"
    def __init__(self, seq_len, pred_len, channels):
        super(DLinearModel, self).__init__()
        self.decompsition = MovingAvg(kernel_size=25, stride=1)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # 시계열 분해
        seasonal_init, trend_init = self.decom_func(x)
        
        # 선형 회귀 적용
        seasonal_output = self.linear_seasonal(seasonal_init.permute(0,2,1)).permute(0,2,1)
        trend_output = self.linear_trend(trend_init.permute(0,2,1)).permute(0,2,1)
        
        return seasonal_output + trend_output

    def decom_func(self, x):
        trend = self.decompsition(x)
        seasonal = x - trend
        return seasonal, trend
```

## 3. 🛡️ Verification (Data Analyst Perspective)
- **Backtesting**: 과거 데이터를 활용하여 '내일'의 결과를 예측하고 실제 결과와 비교하는 워크플로우 필수.
- **Residual Analysis**: 예측값과 실제값의 차이(잔차)가 정규분포를 따르는지 확인하여 모델의 신뢰도 검증.
- **SHAP Integration**: 예측에 가장 큰 기여를 한 '패치'나 '피처'가 무엇인지 시각화.

---
*Inspired by TimesFM and DLinear Research*
