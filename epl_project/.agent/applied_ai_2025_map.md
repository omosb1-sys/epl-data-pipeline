# 🧠 2025 Applied AI Frontier Map (Context & Trajectory)

이 문서는 Adarsh Reganti의 'State of Applied AI 2025' 보고서를 안티그래비티 엔진에 이식한 지식 베이스입니다. 단순한 채팅 기능을 넘어, 추론(Reasoning)하고 과정을 평가(Trajectory)하는 차세대 시스템 구축을 목표로 합니다.

## 🧭 2025 에이전트 핵심 전략 (The 2025 Shift)

| 전략 레이어 | 핵심 기술 | Antigravity 최적화 포인트 | 활용 포인트 |
| :--- | :--- | :--- | :--- |
| **Input (Context)** | Context Engineering | 프롬프트가 아닌 '컨텍스트' 설계 | Lost-in-the-Middle 방어, 그래프 검색 |
| **Model (Reasoning)** | System 2 Thinking | Chain-of-Thought 강제화 (o1/R1) | 복잡한 아키텍처 및 알고리즘 구현 |
| **Application (T-Shape)** | Unit-Specialized Agents | 에이전트의 T자형 전문화 | 법률, 데이터, 시각화 전담 에이전트 |
| **Output (Trajectory)** | Trajectory Challenge | 과정 평가 및 자기 교정 | 실패 원인 분석 및 재시도 전략 |

## 🛠️ 실전 구현 가이드 (2025 Patterns)

### 1. 🔍 컨텍스트 엔지니어링 (Context Engineering)
- **Concept**: 컨텍스트 윈도우의 '어디에' 정보를 배치할지 설계합니다.
- **Lost-in-the-Middle Defense**: 중요한 데이터는 컨텍스트의 최상단과 최하단에 배치하고, 중간 부분에는 압축된 요약본만 넣습니다.
- **Agentic Retrieval**: 단순히 단어로 찾지 않고, 에이전트가 "필요한 정보를 찾기 위한 계획"을 세우고 검색하게 합니다.

### 2. 🛤️ 트래젝토리 평가 (Trajectory-Based Evaluation)
- **Concept**: 에이전트의 '중간 풀이 과정'을 평가합니다.
- **Process**: [문제 인식] -> [도구 선택] -> [실행] -> [결과 검증] 각 단계의 로그를 분석하여, 어느 단계에서 편향이나 오류가 발생했는지 식별합니다.

### 3. 🤖 T자형 에이전트 협업 (T-Shaped Collaboration)
- **Architecture**: 메인 오케스트레이터(System 2)와 전문 서브 에이전트(T-Shaped) 간의 MCP 기반 통신.
- **Efficiency**: 모든 정보를 공유하는 대신, 각 에이전트가 처리한 '정제된 인사이트'만 공유하여 컨텍스트 부하를 줄입니다.

---
## 💡 안티그래비티 2025 업그레이드 체크리스트
1. **System 2 Activation**: 복잡한 요청 시 "한 번 더 생각(CoT)" 루프를 가동했는가?
2. **Context Balancing**: 중요한 정보가 윈도우 중간에 묻히지 않았는가?
3. **Trajectory Check**: 실패 시 결과물만 보지 않고 '과정(Log)'을 분석했는가?

---
*Created by Antigravity (Applied AI 2025 Intel) - 2026.01.21*
