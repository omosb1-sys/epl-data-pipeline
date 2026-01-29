# 🏗️ Anti-gravity Multimodal Engine (Naive & Buoyancy)

이 문서는 Candra Alpin Gunawan의 "Naive Formula" 및 "Signal Buoyancy(ReZero)" 연구를 안티그래비티 프로젝트의 텍스트-이미지 멀티모달 엔진 설계에 이식한 지식 베이스입니다. 최소한의 연산으로 최대의 정보 흐름을 유지하는 고성능 엔진 구축을 목표로 합니다.

## 🧭 엔진 핵심 아키텍처 (The Dual-Force Engine)

| 엔진 컴포넌트 | 핵심 기술 | 역할 | Antigravity 최적화 포인트 |
| :--- | :--- | :--- | :--- |
| **Thrust (추진력)** | Naive Formula (Additive Attention) | 연산량 $O(N^2) \rightarrow O(N)$ 절감 | 텍스트/이미지의 병렬 고속 처리 |
| **Lift (부력)** | ReZero & Jacobian Reg | 레이어별 신호 증폭 및 안정화 | 깊은 모델에서의 정보 소실(중력) 방지 |
| **Core (심장)** | JAX / Equinox | 함수형 최적화 및 JIT 컴파일 | 수학적 무결성 및 Mac(8GB RAM) 고속 학습 |

## 🛠️ 실전 구현 가이드 (JAX-based Patterns)

### 1. 🚀 Naive Additive Attention (연산의 안티그래비티)
- **Concept**: 행렬 곱셈 대신 `Q + K` 기반의 선형 결합 사용.
- **JAX Implementation**:
```python
def naive_lift_attention(q, k, v, alpha):
    # Q+K 기반의 가벼운 추진력
    thrust = jax.nn.sigmoid(q + k) * v
    # ReZero 부력 제어: 초기 alpha=0 (무중력 상태 유지)
    return jnp.tanh(alpha) * thrust
```

### 2. 🏥 Signal Buoyancy (신호의 안티그래비티)
- **Zero-Gravity Warmup**: 학습 초기에는 레이어의 모든 파라미터를 투명하게(Transparent) 설정하여 신호가 바닥까지 저항 없이 흐르게 합니다.
- **Jacobian Stability**: `jax.jacfwd`를 사용하여 입력 변화에 대한 출력의 불필요한 폭주를 제어하고 균형을 유지합니다.

### 3. 🖼️ Multimodal Balancing (Text & Image)
- **Text (Sequence Buoyancy)**: 긴 문장에서 초기 토큰 정보가 가라앉지 않도록 부력 계수(alpha)를 시퀀스 위치에 따라 동적으로 조절합니다.
- **Image (Entropy Repulsion)**: 이미지의 특정 부분에만 함몰되지 않도록 엔트로피 척력을 가하여 엔진이 전체 맥락을 골고루 띄워 올리게 합니다.

---
## 💡 안티그래비티 엔진 학습 워크플로우
1. ** 시동 (Ignition)**: 모든 `alpha`를 0으로 설정하여 무중력 상태로 데이터 패스를 탐색합니다.
2. ** 가속 (Thrust)**: Naive Formula를 통해 텍스트와 이미지 데이터를 고속으로 결합합니다.
3. ** 수평 유지 (Leveling)**: Jacobian 정규화를 통해 엔진의 진동(Gradient Instability)을 최소화합니다.

---
*Created by Antigravity (Candra's Naive & Buoyancy Engine) - 2026.01.21*
