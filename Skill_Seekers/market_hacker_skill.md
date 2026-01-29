# Market Hacker Skill (Lee Hyo-yeon Protocol)

이 스킬은 데이터를 단순 분석하는 수준을 넘어, 시장의 결핍을 포착하고 사용자(기획자)의 통찰력을 증강하기 위한 특수 프롬프트 및 로직 세트입니다.

## 1. Context Generator (맥락 생성기)
데이터 분석 결과 전달 시, 다음 질문에 대한 답을 반드시 포함한다.
- **Why Now?**: 왜 이 데이터가 지금 이 시점에 가장 중요한가? (EPL 이적 시장 개장, K-리그 개막, 연휴 시즌 등)
- **The Thread**: 이 지표가 과거의 어떤 흐름과 연결되어 있으며, 미래의 어떤 갈증을 예고하는가?

## 2. Shadow KPI Finder (잠재적 결핍 탐지)
사용자가 명시하지 않은 필터링 항목을 제안한다.
- **Fatigue Index**: 단순 활동량이 아닌, 누적된 '피로도'로 인한 부상 위험 및 경기력 저하 추적.
- **Silence Analysis**: 게시판이나 뉴스에서 언급되지 않지만, 지표상으로 급격히 변하고 있는 '소리 없는 변화' 포착.
- **Sentiment Gap**: 통계적 우수성과 팬들의 체감(Sentiment) 사이의 괴리 분석.

## 3. Curator's Eye Prompt (전문가 필터)
AI가 수천 개의 데이터 중 '감도 높은' 3가지를 선별하는 기준:
1. **Uniqueness**: 기존 분석가들이 흔히 놓치는 지점인가? (e.g., 하프스페이스 점유율보다 '전환 시 첫 패스의 방향')
2. **Actionability**: 기획자가 즉시 비즈니스/전술에 적용할 수 있는가?
3. **Story Potential**: 이 데이터로 팬들에게 매력적인 서사를 들려줄 수 있는가?

## 4. Execution Fast-Track (초고속 실행 코드)
인사이트 도출 시 즉시 사용할 수 있는 Python Snippet 예시:
```python
# 예: 특정 선수의 잠재적 결핍(체력 고갈) 시각화
import pandas as pd
import matplotlib.pyplot as plt

def analyze_latent_fatigue(df: pd.DataFrame, player_id: str) -> None:
    # 단순 주행 거리가 아닌, 고강도 스프린트 횟수의 급감 지점 포착
    ...
```

## 5. Decision Support (의사결정 보조)
기획자가 "이거 해볼까요?"라고 물었을 때의 답변 구조:
- **Impact**: 예상되는 시장의 파급력
- **Risk**: 놓치고 있는 리스크 (Red Team Attack)
- **Hacking Point**: 경쟁자가 생각하지 못한 '비틀기(Hacking)' 포인트

---
*Applied by Antigravity (Lee Hyo-yeon Protocol Upgrade v1.0)*
