---
description: AI 기반 초현실적 데이터 시각화 시스템
---

# AI-Powered Visualization Engine

## 목적
분석 결과를 '무중력' 컨셉의 혁신적 비주얼로 변환

## 시각화 레벨

### Level 1: 기본 차트 (현재)
```python
import matplotlib.pyplot as plt
sns.histplot(data)
```

### Level 2: 인터랙티브 3D (추가)
```python
import plotly.graph_objects as go

# 3D 산점도로 변수 간 관계 표현
fig = go.Figure(data=[go.Scatter3d(
    x=df['total_shots'],
    y=df['success_rate'],
    z=df['win'],
    mode='markers',
    marker=dict(
        size=8,
        color=df['win'],
        colorscale='Viridis',
        showscale=True
    )
)])
fig.update_layout(
    title='K-리그 승패 예측 3D 공간',
    scene=dict(
        xaxis_title='슈팅 수',
        yaxis_title='성공률',
        zaxis_title='승리 여부'
    )
)
fig.show()
```

### Level 3: AI 생성 인포그래픽 (최고급)
```python
from generate_image import create_infographic

# 분석 결과를 AI가 자동으로 인포그래픽화
insights = {
    'accuracy': 0.67,
    'top_feature': 'total_shots',
    'causal_effect': -0.172
}

create_infographic(
    data=insights,
    style='anti-gravity',  # 무중력 테마
    format='floating_cards',  # 떠있는 카드 레이아웃
    animation=True  # 애니메이션 효과
)
# → output/ai_infographic.mp4 생성
```

## 자동 적용 시점
- 모든 분석 완료 후 자동으로 3가지 버전 생성:
  1. 정적 이미지 (PNG)
  2. 인터랙티브 HTML (Plotly)
  3. AI 인포그래픽 (MP4)

## 예시 출력
```
📊 분석 결과 시각화 완료!
   ├── basic_chart.png (기본)
   ├── interactive_3d.html (인터랙티브) ← 브라우저에서 열기
   └── ai_infographic.mp4 (AI 생성) ← SNS 공유용
```
