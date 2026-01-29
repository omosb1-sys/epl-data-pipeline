# ⚖️ Precision Intelligence Map (CUAD & Legal Principle)

이 문서는 Microsoft Foundry의 CUAD(Contract Understanding Atticus Dataset) 분석 통찰을 안티그래비티 프로젝트의 문서 지능(Document Intelligence)에 이식한 지식 베이스입니다. 복잡한 텍스트에서 99.9%의 정확도로 핵심 정보를 추출하고 관리하기 위해 이 가이드를 따릅니다.

## 🧭 고정밀 문서 분석 전략 (Precision & Traceability)

| 컴포넌트 | 핵심 기술 | Antigravity 적용 이점 | 활용 포인트 |
| :--- | :--- | :--- | :--- |
| **Precision Extraction** | CUAD 벤치마크 기반 추출 | 독소 조항 및 핵심 의무 사항 식별 | 선수 계약서, 리그 규정집 분석 |
| **Semantic Refactoring** | 문서 리팩토링 (Clause-level) | 문맥 파괴 없는 정밀 조항 업데이트 | 구단 정책 및 법률 문서 수정 |
| **Traceability** | 근거 기반 추론 (Evidence-Link) | AI 판단의 신뢰도 및 투명성 확보 | 상무님 보고용 근거 자료 제시 |

## 🛠️ 실전 구현 가이드 (Document Engineering)

### 1. 🔍 고정밀 추출 (Extraction Mode)
- **Concept**: 단순 키워드 검색이 아닌, 문맥(Context)을 이해하여 특정 '의무(Obligation)'나 '권리(Right)'를 추출합니다.
- **Recipe**: 에이전트에게 "CUAD 가이드라인에 따라 이 계약서의 'Termination for Convenience' 조항을 추출하고 표준안과 대조해줘"라고 명령하세요.

### 2. 📝 문서 리팩토링 (Refactoring Mode)
- **Concept**: 코드를 고치듯 문서의 특정 섹션을 정규화된 양식으로 교체합니다.
- **Recipe**: `.agent/templates/`에 법 표준 양식을 두고, 에이전트가 기존 문서를 읽어 해당 양식에 맞춰 '시맨틱 업데이트'를 수행하게 합니다.

### 3. 🔗 추적성 및 승인 (Approval Loop)
- **Principle**: AI는 조연(Pilot), 사용자는 감독(Captain).
- **Workflow**: [분석 수행] -> [판단 근거 제시(Link)] -> [사용자 리뷰] -> [최종 승인/반영]의 4단계 루프를 강제합니다.

---
## 💡 안티그래비티 법률/문서 인텔리전스 루프
1. **Bulk Ingestion**: 수백 개의 계약서를 `data/contracts/`에 로드합니다.
2. **Precision Audit**: `@DocumentAgent`가 CUAD 기법으로 전수 조사를 실시합니다.
3. **Diff Report**: 표준안과 다른 조항을 `output/legal_reports/`에 리포트로 생성합니다.
4. **Action**: 사용자의 지시에 따라 특정 조항을 일괄 리팩토링합니다.

---
*Created by Antigravity (CUAD Precision Intelligence) - 2026.01.21*
