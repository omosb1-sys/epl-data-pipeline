---
name: precision-extractor
description: 비정형 데이터(뉴스, 리뷰 등)를 정형 데이터(JSON, 태그, 점수)로 정밀 추출하는 Hyel AI 스타일 파이프라인
user-invocable: true
---

# 💎 Precision Extractor: 비정형 데이터 정밀 자산화 지침

본 스킬은 비정형 데이터를 나중에 쿼리가 가능한 '데이터 자산'으로 변환하며, `GEMINI.md`의 **Rule 8.5 (LLM-Ready Extraction)**를 준수합니다.

## 🎯 1. 데이터 자산화 원칙 (Assetization)
1.  **Structuring over Summary**: 단순 요약보다 나중에 검색 및 집계가 가능한 **JSON/TOON 스키마**로 변환하는 것을 최우선으로 합니다.
2.  **Entity Resolution**: 텍스트 내 주요 개체(선수, 팀, 부상 부위 등)를 고유 ID와 매핑하여 데이터베이스 정합성을 확보합니다.
3.  **Confidence Tagging**: 추출된 정보마다 확신도 점수를 부여하여, Rule 14 (CAE) 프로토콜의 판단 근거로 활용합니다.

## 🌀 2. 재귀적 추출 (RLM Chain - Rule 36)
1.  **Recursive Slicing**: 방대한 기사나 논문을 처리할 때, 하위 섹션 단위로 나누어 정밀 추출 후 이를 계층적으로 합성(Synthesis)합니다.
2.  **Deduplication Log**: 동일한 뉴스 소스가 여러 개일 경우, 정보의 신선도와 상세도를 기준으로 중복을 제거하고 가장 완벽한 'Gold Record'를 생성합니다.

## 📝 3. 워크플로우 (Hyel AI Style)
1.  **Scan & Clean**: 특수문자 및 광고성 텍스트를 제거하여 노이즈를 최소화합니다.
2.  **Extract & Map**: 핵심 메타데이터를 추출하고 프로젝트의 기존 스키마(`fixtures`, `transfers` 등)에 매핑합니다.
3.  **Verify & Distill**: 추출 결과를 다른 상충하는 데이터와 대조 검증 후, 핵심 인사이트만 장기 메모리에 저장합니다.

---
*Updated by Antigravity (Advanced Extraction & Assetization Lab) - 2026.01.26*
