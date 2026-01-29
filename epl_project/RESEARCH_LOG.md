# 🧪 Antigravity AI Research Log

이 파일은 Antigravity 엔진의 모델 학습, 실험 및 아키텍처 결정을 기록하는 전문 연구 로그입니다. Candra Alpin Gunawan의 'Research Log' 원칙을 준수합니다.

---

## [2026-01-21] Phase 1: Advanced Transformer Architecture Integration

### 1. 실험 배경
*   **목적**: EPL 데이터 파이프라인의 예측 엔진 고도화 및 8GB RAM 환경에서의 모델링 안정성 확보.
*   **참조 소스**: Candra Alpin Gunawan의 GAU/MQA 베스트 프랙티스 및 DETR 수렴 연구.

### 2. 아키텍처 변경 사항
*   **GAU (Gating Attention Units)** 도입: 시계열 특징 추출의 성능 향상.
*   **MQA (Multi-Query Attention)** 도입: 추론 속도 최적화 및 수렴 안정성 확보.
*   **SwiGLU Activation**: 기존 ReLU/GELU 대비 풍부한 표현력 제공.
*   **Rezero Gate**: 잔차 연결 초기화 시 zero-gravity 상태를 유지하여 신호 소실 방지.

### 3. 기술 논평
*   **MQA vs MHA**: NLP 태스크에서 검증된 MQA는 EPL의 경기 시퀀스 데이터 분석에서도 파라미터 효용성이 높을 것으로 기대됨.
*   **Convergence Strategy**: GAU 적용 시 수렴 속도가 느려질 경우 Multi-Head Attention으로의 유연한 회항 전략을 `rules.md`에 명시함.

### 4. 다음 단계
*   `train_ai_model.py`에 GAU/MQA 기반 레이어 실제 구현 및 벤치마킹.
*   학습 곡선(Loss Curve) 시각화 및 로그 기록 자동화.

---
*Logged by Antigravity (@DataAgent)*
