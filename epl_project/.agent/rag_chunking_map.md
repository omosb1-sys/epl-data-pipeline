# 🧩 Antigravity RAG Chunking Strategy (Simone's Principle)

이 문서는 Simone Fuscone이 제시한 "RAG를 위한 4대 청킹(Chunking) 방법론"을 안티그래비티 프로젝트에 최적화하여 정리한 지식 베이스입니다. 대규모 뉴스 데이터나 전술 문서 처리 시 최적의 검색 품질을 확보하기 위해 이 가이드를 따릅니다.

## 🗺️ 4대 청킹 전략 요약

| 전략 | 방법 | 장점 | 단점 | Antigravity 활용 사례 |
| :--- | :--- | :--- | :--- | :--- |
| **Fixed-Size** | 500토큰 등 고정 크기 분할 | 빠름, 구현 단순 | 문맥 단절 위함 | 대량의 실시간 뉴스 스트리밍 |
| **Recursive** | \n, 마침표 기반 단계적 분할 | 문장/문단 보존 탁월 | 로직 복잡성 | 정교한 전술 칼럼 및 매뉴얼 |
| **Document-Based** | Markdown 헤더, JSON 구조 인식 | 구조적 무결성 유지 | 일관된 포맷 필요 | `SPEC.md`, `agents.md` 기반 분석 |
| **Semantic** | 의미 변화 지점 AI 감지 | 검색 정확도 최상 | 속도 느림, 비용 발생 | 고차원적 전술-데이터 융합 분석 |

## 🛠️ 실전 적용 가이드 (Best Practices)

### 1. 📏 Fixed-Size Chunking (Overlap 전략)
- **언제 쓰나?**: 텍스트의 구조보다 **속도**가 중요할 때.
- **Tip**: 앞뒤 청크를 50~100자 정도 겹치게(Overlap) 설정하여 중간에 잘린 정보를 보완하세요.

### 2. 🔄 Recursive Character Splitting
- **언제 쓰나?**: 문맥 유지가 필수적인 **보도자료나 칼럼** 처리 시.
- **Tip**: `\n\n` -> `\n` -> `.` 순으로 분할 우선순위를 정해 문정 무결성을 확보하세요.

### 3. 🏗️ Document-Based Splitting
- **언제 쓰나?**: **기술 명세서나 규정집**처럼 계층 구조가 명확할 때.
- **Tip**: Markdown의 `#`, `##` 헤더를 구분자로 활용해 주제별 덩어리를 만드세요.

### 4. 🧠 Semantic Chunking (AI-Powered)
- **언제 쓰나?**: 복잡한 인과관계가 얽힌 **심층 분석** 시.
- **Tip**: 임베딩 유사도가 급격히 변하는 지점(Split point)을 포착하여 '완결된 아이디어' 단위로 쪼개세요.

---
## 💡 안티그래비티 학습 루프
1. **정제(Cleaning)**: 청킹 전 노이즈 데이터를 먼저 제거합니다.
2. **테스트(A/B Test)**: 각 방식이 실제 `get_relevant_context` 성능에 어떤 영향을 주는지 비교합니다.
3. **증류(Distillation)**: 유용한 청크는 `team_memory.json`으로 영구 보관합니다.

---
*Created by Antigravity (Simone's RAG Architecture) - 2026.01.21*
