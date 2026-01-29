# [Skill] Solar AI 데이터 분석 및 해석 (물어봐)

## 1. 개요
Upstage의 Solar Open 100B 모델을 활용하여 복잡한 데이터 분석 결과(통계치, 그래프 의미 등)를 전문가 수준으로 해석해주는 기능입니다.

## 2. 관련 파일
*   핵심 로직: [my_analyst.py](cci:7://file:///Users/sebokoh/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D%EC%97%B0%EC%8A%B5/%EB%8D%B0%EC%9D%B4%EC%BD%98/k%EB%A6%AC%EA%B7%B8%EB%8D%B0%EC%9D%B4%ED%84%B0/%EB%A6%AC%EA%B7%B8%EB%8D%B0%EC%9D%B4%ED%84%B0/my_analyst.py:0:0-0:0)

## 3. 사용 방법 (Usage)
분석 중인 Python 환경에서 다음과 같이 호출합니다.

```python
from my_analyst import 물어봐

# 예시: 통계 결과 해석 요청
물어봐("ANOVA 결과 p-value가 0.1110이 나왔어. 의미를 설명해줘.")