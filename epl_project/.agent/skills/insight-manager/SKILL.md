---
name: insight-manager
description: 글로벌 기술 인사이트(LinkedIn, X, RSS, Newsletter 등) 수집, 요약 및 시스템 자동 적용 관리
user-invocable: true
---

# 🌐 글로벌 기술 인사이트 매니저 (Intelligence Hub)

본 스킬은 글로벌 오피니언 리더들로부터 최신 기술 인사이트를 수집하고, 이를 안티그래비티의 지식 베이스에 자산화(Assetization)하는 프로토콜입니다.

## 📡 1. 지능형 멀티 채널 수집 (Rule 19.1 연계)
1.  **LinkedIn/X/RSS**: `firecrawl` 또는 브라우저 툴을 사용하여 지정된 핵심 인물(Karpathy, Chip Huyen 등)의 포스트를 스캔합니다.
2.  **Newsletter 파싱**: 수신된 Lenny's Newsletter 등을 구조화된 JSON(TOON)으로 변환하여 저장합니다.
3.  **YouTube Summarization**: 최신 테크 영상의 핵심 내용을 추출하여 `project_knowledge.json`에 시계열로 누적합니다.

## 🌀 2. 재귀적 지식 증류 (RLM Protocol - Rule 36)
1.  **Context Consolidation**: 수집된 방대한 정보를 재귀적으로 요약하여, 전체 맥락은 유지하되 데이터 밀도는 최고로 높인 **'Insight Blocks'**를 생성합니다.
2.  **Distillation Engine**: 수집된 뉴스 중 시스템 적용 가치가 있는 것만 선별하여 `rules.md`나 관련 스킬의 업데이트를 제안합니다.

## 🚀 3. 운영의 탁월함 (SRE Protocol - Rule 37)
1.  **Insight Health Check**: 수집된 정보의 신뢰성과 신선도를 주기적으로 검증하여, 낡은 정보가 시스템에 영향을 주지 않도록 관리합니다.
2.  **Automated Cleanup**: 보고가 완료된 원천 데이터는 7일 후 자동으로 삭제하여 Mac의 8GB RAM 환경을 보호합니다. (Rule 15.4 연계)

---
*Updated by Antigravity (Global Tech Intelligence Hub) - 2026.01.26*
