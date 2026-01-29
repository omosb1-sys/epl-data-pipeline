# 🎓 Data Analyst Growth Roadmap (Antigravity Curated)

이 문서는 사용자가 단순한 '데이터 작업자'를 넘어 '엔지니어링 역량을 갖춘 시니어 분석가'로 성장하기 위한 로드맵입니다. LinkedIn 인기 리포지토리와 현재 진행 중인 프로젝트를 결합했습니다.

## 1. 🚀 필수 학습 리포지토리 (LinkedIn TOP 20 기반)

| 우선순위 | 리포지토리 | 학습 목표 | 관련 프로젝트 |
| :--- | :--- | :--- | :--- |
| **0순위** | `DataExpert-io/data-engineer-handbook` | 대용량 데이터 ETL, 데이터 모델링 원리 이해 | 🚗 크몽 프로젝트 (240만 행) |
| **1순위** | `GokuMohandas/Made-With-ML` | 모델 배포, 모니터링, 데이터 검증(MLOps) | ⚽ EPL 실시간 예측 시스템 |
| **2순위** | `microsoft/ai-agents-for-beginners` | AI 에이전트 설계 및 자동화 로직 | 🤖 안티그래비티 고도화 |
| **3순위** | `rasbt/LLMs-from-scratch` | LLM의 작동 원리 이해 (프롬프트 한계 극복) | 🛡️ 시니어 분석가 페르소나 |
| **4순위** | `jakevdp/PythonDataScienceHandbook` | Pandas/Numpy 성능 최적화 및 시각화 정석 | 📊 상무님 보고용 시각화 |

---

## 2. 🛠️ 프로젝트별 업그레이드 전략

### 🚗 크몽 프로젝트 (차량 데이터 분석)
*   **Engineering First**: 단순 `pandas` 분석이 아닌 `Polars` + `DuckDB` 기반의 **고속 파이프라인**을 정교화합니다.
*   **Data Validation**: `kmong_engine.py`에 적용된 `DataQualityGuard`를 발전시켜, 데이터 소스가 바뀌어도 에러가 나지 않는 강건한 코드를 작성하는 훈련을 합니다.
*   **Reporting Engineering**: PPT/PDF 리포트 생성을 자동화하여 업무 생산성을 5배 이상 높이는 기술(Python-pptx 활용)을 마스터합니다.

### ⚽ EPL 실시간 프로젝트 (예측 알고리즘)
*   **Real-time Infrastructure**: `pathwaycom/llm-app` 스타일의 실시간 스트리밍 데이터를 처리하는 감각을 익힙니다. 
*   **Model Observability**: 예측값이 왜 그렇게 나왔는지 SHAP 차트를 통해 설명력을 확보(XAI)하는 연습을 합니다.

---

## 3. 🎯 시니어 분석가의 태도 (Growth Mindset)
*   **Don't Just Report, Recommend**: 데이터 수치를 보여주는 것에 그치지 말고, "그래서 비즈니스적으로 무엇을 해야 하는가(Actionable Insight)"를 제언하는 연습을 하세요.
*   **Refactor Often**: 코드를 짠 후 반드시 10% 더 효율적인 방법(메모리 절약, 가독성 향상)이 없는지 고민하고 리팩토링하세요.
*   **Be the Architect**: 도구(Copilot, Cursor, Antigravity)에게 휘둘리지 말고, 전체 아키텍처를 설계하고 도구에게는 '구현'을 시키는 설계자(Architect)의 관점을 유지하세요.

---
*Created by Antigravity (Gemini 3) - Your Strategic Partner in Career Growth*
