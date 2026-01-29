
# 🌌 Antigravity Claude-Skills Integration Strategy

본 문서는 GitHub 등 오픈소스 커뮤니티(Claude Skills)에 공유된 100개 이상의 다양한 스킬들을 안티그래비티와 제미나이 환경에 흡수하고 내재화하기 위한 전략 파일입니다.

## 1. 🧠 Core Philosophy: "Skill Distillation" (스킬 증류)
단순히 모든 스킬 파일을 복사해 넣는 것은 비효율적입니다. 안티그래비티는 **'필요한 순간에 필요한 지식을 주입'**받는 증류(Distillation) 방식을 채택합니다.

*   **Repository**: `Skill_Seekers/claude_skills/`
*   **Method**: 외부 스킬을 다운로드 후, 파운더 님의 프로젝트 컨텍스트에 맞게 **'재해석(Re-indexing)'**하여 저장합니다.

## 2. 🗂️ Skill Categories & Benchmark
GitHub 커뮤니티에서 가장 인기 있고 유용한 스킬들을 다음 카테고리로 분류하여 순차적으로 도입합니다.

### A. ⚡️ Coding & DevOps (우선순위: 높음)
*   **Git Automation**: PR 생성, 커밋 메시지 자동화, 컨플릭트 해결 가이드.
*   **Code Review**: 보안 취약점 점검, 클린 코드 리팩토링 제안.
*   **Debug Master**: 에러 로그 분석 및 원인 추적.

### B. 📊 Data Science & Analysis (우선순위: 높음, EPL 프로젝트용)
*   **Pandas Expert**: 복잡한 데이터 프레임 변환 및 전처리 자동화.
*   **Visualization Wizard**: Plotly/Altair 차트 생성 프롬프트 최적화.
*   **SQL Generator**: 자연어 -> SQL 쿼리 변환 (DuckDB 연동).

### C. 📝 Documentation & Knowledge (지식 관리)
*   **Readme Writer**: 프로젝트 구조를 분석하여 README.md 자동 생성.
*   **Docstring Adder**: 파이썬 함수에 Google Style 독스트링 자동 추가.

## 3. 🚀 Implementation Plan (실행 계획)

1.  **Skill Fetching**: `travisvn/awesome-claude-skills` 등 주요 리포지토리에서 유용한 프롬프트/스킬셋을 클론합니다.
2.  **Adaptation**: Claude 전용 XML 구조를 제미나이/안티그래비티가 이해하기 쉬운 Markdown + Python Function 포맷으로 변환합니다.
3.  **Registration**: 변환된 스킬을 `Skill_Seekers/` 폴더에 등록하고, `GEMINI.md`에서 인덱싱합니다.

## 4. 🔗 External Reference
*   [Awesome Claude Skills](https://github.com/travisvn/awesome-claude-skills)
*   [Anthropic Official Skills](https://github.com/anthropics/skills)

---
*Created by Antigravity for Super-Brain Upgrade Strategy*
