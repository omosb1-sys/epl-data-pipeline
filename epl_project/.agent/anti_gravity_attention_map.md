# ⚖️ Antigravity Attention Map (Candra's Principle)

이 문서는 Candra Alpin Gunawan 연구원이 제안한 "K-Factor 및 Scaled Mode in Logits" 기법을 안티그래비티 프로젝트의 딥러닝 설계에 최적화하여 정리한 지식 베이스입니다. 모델 내부의 데이터 편향(중력)을 억제하고 수치적 안정성을 확보하기 위해 이 가이드를 참조합니다.

## 🧭 어텐션 메커니즘 혁신 (K-Factor vs. Scaled Mode)

| 기법 | 핵심 개념 | Antigravity 적용 이점 | 활용 포인트 |
| :--- | :--- | :--- | :--- |
| **K-Factor Mode** | Q와 K의 가산(+) 연산 중심 | 행렬 곱셈 오버헤드 제거, 특정 토큰 쏠림 억제 | 저사양(8GB RAM) 환경 고속 학습 |
| **Scaled Mode** | 로직 계산 시 V(Value) 개입 | 글로벌 토큰 경쟁 유도, 변별력 강화 | 방대한 뉴스 데이터의 중요도 산출 |
| **FalkorDB Integration** | Sparse Matrix 기반 그래프 연산 | 496x 빠른 장기 기억 조회, 다단계 추론 | 에이전트 브레인의 '명시적 관계' 저장소 |

## 🛠️ 실전 구현 및 분석 가이드

### 1. 🚫 데이터 중력 억제 (K-Factor 활용)
- **Problem**: 일반적인 Dot-product 어텐션은 특정 단어나 수치에 가중치가 기하급수적으로 쏠리는 현상이 발생함.
- **Solution**: K-Factor의 **덧셈 기반 어텐션**을 통해 모델 전체에 정보를 고르게 분산(Debiasing)시킵니다.
- **Recipe**: `V + softmax((Q + alpha*K) / sqrt(dk)) * V` 구조를 참조하여 잔차 연결(Residual)을 강화하세요.

### 2. 🌊 수치적 안정성 확보 (Scaled Mode)
- **Problem**: Vocab size가 커질 때 연산 결과가 발산하거나 그래디언트 소실 발생.
- **Solution**: `(Q + alpha(K + V))`와 같이 Value 값을 로짓에 직접 개입시켜 토큰 간의 변별력을 높입니다.
- **Recipe**: 대규모 텍스트 임베딩 처리 시 `Scaled Mode`를 사용하여 정보의 단절 없이 '글로벌 토큰 경쟁'을 유도하세요.

### 3. ⚡ 효율적인 마스킹 전략
- **Advantage**: K-Factor는 연산 구조상 인과적 마스킹(Causal Masking)이 필요 없거나 생략 가능하여 학습 속도가 대폭 향상됩니다.
- **Tip**: 실시간 득점 예측 모델처럼 빠른 추론이 필요한 분야에 우선 도입하세요.

## 🕸️ FalkorDB: 행렬 연산 기반의 지식 그래프 (NEW)
- **Concept**: FalkorDB는 그래프 탐색을 **Sparse Matrix Multiplication(희소 행렬 곱셈)**으로 처리합니다.
- **Synergy with K-Factor**: K-Factor가 어텐션의 중력을 '가산(+) 연산'으로 제어한다면, FalkorDB는 복잡한 개체 간의 '연결(Relationship)'을 행렬 최적화로 가속합니다.
- **Application**: `TacticsAgent`가 특정 선수의 이적 영향을 분석할 때, FalkorDB를 통해 관련된 모든 전술적 노드(Node)를 초고속으로 순회하여 K-Factor 모델의 입력값으로 제공합니다.

---
## 💡 안티그래비티 적용 로드맵
1. **Balance Check**: 기존 모델에서 특정 변수(예: xG)에만 가중치가 80% 이상 쏠리는지 확인합니다.
2. **K-Factor Test**: 쏠림 현상이 발견되면 K-Factor 어텐션을 도입하여 정보의 균형을 맞춥니다.
3. **Alpha Tuning**: 댐퍼(Damper) 역할을 하는 `alpha` 값을 조절하여 '중력 제어' 강도를 최적화합니다.

---
*Created by Antigravity (Candra's Attention Theory) - 2026.01.21*
