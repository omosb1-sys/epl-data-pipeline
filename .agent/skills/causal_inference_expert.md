
# 📈 SKILL: Causal Inference Expert (Microsoft DoWhy/CausalImpact)

> **"Correlation is not Causation."**  
> 이 스킬은 단순한 상관관계를 넘어, 데이터 속에 숨겨진 **'진짜 원인(Causal Effect)'**을 밝혀내는 인과 추론 방법론을 제공합니다.

## 1. Core Principles (핵심 원칙)
*   **Explicit Assumptions (가정의 명시화)**: 모든 분석 전, 변수 간의 관계(Graph)를 먼저 그린다. (DAG)
*   **Refutation First (반증 우선)**: "내 결론이 틀렸을 수도 있다"는 전제로, 반드시 반박 테스트(Placebo, Subset)를 통과해야만 결과를 인정한다.
*   **Method Agnostic**: 단일 모델에 의존하지 않고, 데이터 특성에 따라 `Propensity Score`, `Linear Regression`, `Machine Learning` 등 최적의 추정기를 선택한다.

## 2. Refutation Checklist (반증 체크리스트)
인과 추론 결과보고서에는 반드시 다음 3가지 테스트 통과 여부를 명시해야 한다.
1.  **Placebo Treatment**: "가짜 약을 줬는데도 효과가 있는가?" (있으면 모델 기각)
2.  **Random Common Cause**: "무작위 변수를 추가해도 결과가 유지되는가?"
3.  **Data Subset**: "데이터 일부를 빼도 결론이 같은가?"

## 3. Recommended Libraries & Snippets
*   **Library**: `dowhy` (Microsoft), `causalimpact` (Google - TimeSeries)

### 3.1 Basic DoWhy Workflow (The 4-Step)
```python
import dowhy
from dowhy import CausalModel

# Step 1: Model (Define the Graph)
model = CausalModel(
    data=df,
    treatment='marketing_campaign',
    outcome='sales',
    common_causes=['seasonality', 'market_trend']
)

# Step 2: Identify (Can we estimate it?)
identified_estimand = model.identify_effect()

# Step 3: Estimate (Calculate the effect)
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)

# Step 4: Refute (Challenge the Result)
refute_results = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="refute_estimate" # Placebo, Random Cause etc.
)
print(refute_results)
```

### 3.2 Google CausalImpact (Time-Series)
*   **Use Case**: "이벤트(광고, 정책 변경) 전후의 효과를 측정하고 싶을 때"
*   **Requirement**: 이벤트 이전(Pre-period) 데이터가 충분해야 하며, 이벤트 영향을 받지 않는 대조군(Control Metric)이 있으면 좋다.

```python
from causalimpact import CausalImpact

# Data: [Response, Control1, Control2...]
pre_period = [0, 69]
post_period = [70, 100]

ci = CausalImpact(df, pre_period, post_period)
print(ci.summary())
ci.plot()
```
