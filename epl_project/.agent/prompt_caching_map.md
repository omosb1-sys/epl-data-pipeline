# 🚀 Antigravity Prompt Caching Map (Efficiency Principle)

이 문서는 "프롬프트 캐싱(Prompt Caching)" 기술을 안티그래비티 프로젝트에 적용하여 비용을 90% 절감하고 응답 속도를 극대화하기 위한 지식 베이스입니다. 대규모 전술 데이터와 사용자 히스토리를 효율적으로 처리하기 위해 이 가이드를 따릅니다.

## 🧭 캐싱 전략 요약 (Cost & Latency Optimization)

| 캐싱 대상 | 내용 | 효과 | Antigravity 활용 포인트 |
| :--- | :--- | :--- | :--- |
| **System Rules** | `rules.md`, `agents.md` 등 고정 규칙 | 비용 절감 | 모든 질의마다 공통으로 적용되는 에이전트 지침 |
| **Fixed Context** | 구단 창단 이력, 감독 철학, 경기장 정보 | 속도 개선 | 변하지 않는 EPL 기본 지식 베이스 |
| **User History** | 이전 분석 결과, 사용자의 선호 팀/선수 | 맞춤형 강화 | `st.session_state` 및 장기 기억(`team_memory.json`)과 연동 |
| **Model Instructions** | 분석 엔진(`UltimateAnalyticEngine`) 가이드 | 정확도 유지 | 복잡한 수치 계산 및 시각화 로직 지침 |

## 🛠️ 실전 적용 가이드 (Best Practices)

### 1. 🏗️ 프롬프트 아키텍처 (Layered Design)
- **Static First**: 절대 변하지 않는 지침(System Prompt)을 가장 앞부분에 배치하여 캐시 적중률(Hit Rate)을 높이세요.
- **Sequential Context**: [시스템 규칙] -> [구단/전술 정보] -> [어제의 경기 결과] -> [사용자의 현재 질문] 순으로 배치하여 변경되는 부분만 새로 계산하게 합니다.

### 2. ⚡ 실시간성 확보 (Real-time Feedback)
- **Problem**: 딥러닝 분석 결과에 대한 코멘트를 생성할 때 지연 시간이 발생함.
- **Solution**: 분석 로직과 지침을 캐싱하여, 새로운 수치 데이터가 유입되는 즉시 0.1초 이내에 인사이트를 출력하도록 설계합니다.

### 3. 📉 운영 비용 최적화 (Cost Saving)
- **Insight**: 프롬프트 캐싱을 통해 Claude 3.5나 Gemini 3 등의 고성능 모델 사용 비용을 최대 90%까지 줄일 수 있습니다.
- **Action**: 절감된 비용 리소스를 더 정교한 '반중력 어텐션(K-Factor)' 연산이나 고화질 시각화 위젯 생성에 재투자하세요.

---
## 💡 안티그래비티 운영 루프
1. **Identify**: 반복적으로 사용되는 긴 텍스트(전술 매뉴얼 등)를 식별합니다.
2. **Cache**: 에이전트가 해당 텍스트를 캐싱된 상태로 읽도록 프롬프트를 구성합니다.
3. **Analyze**: 지연 시간 단축 효과를 `@SecurityAgent`가 모니터링하여 최적의 캐싱 타이밍을 결정합니다.

---
*Created by Antigravity (Prompt Caching Architecture) - 2026.01.21*
