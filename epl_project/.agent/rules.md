# EPL Project Rules & Tips

이 파일은 Claude Code 해커톤 우승자의 팁, Lior Alex의 의도 분석, K-Dense의 과학적 기술, 그리고 Manus의 컨텍스트 관리 방식을 통합하여 사용자님의 Mac(8GB RAM) 환경에 최적화한 절대 규칙입니다.

## 1. 🧠 Context Management (컨텍스트 관리)
*   **Minimal Attachment**: 작업 시 불필요한 파일은 @Agent 호출에서 제외한다.
*   **Focus Mode**: 에디터에 열려있는 탭을 현재 작업 중인 파일로 최소화한다.
*   **Small Chunks**: 요청을 논리적으로 완결된 작은 단위로 쪼개서 수행한다.

## 2. 🎭 Agent Specialization (에이전트 역할 분담)
*   **Planning Agent**: 전체 아키텍처와 로직 설계 담당.
*   **Coding Agent**: 실제 기능 구현 및 버그 수정 담당.
*   **Review Agent**: 작성된 코드의 성능과 보안성을 검토.
*   **Specialized Personas**: `@TacticsAgent`, `@DataAgent` 등 `agents.md`에 정의된 페르소나를 호출한다.

## 3. 📝 Coding Convention & Safety (코딩 컨벤션 및 안전)
*   **Strict Typing**: 모든 Python 코드에는 Type Hinting을 필수 적용한다.
*   **Docstring Protocol**: Google Style Docstring을 사용하여 역할을 명확히 기술한다.
*   **Test-Driven Execution**: 수정 후에는 반드시 `verify_ai_engine.py` 등 테스트를 실행한다.

## 4. 🔄 Workflow: Human-in-the-loop (Plan Before Write)
*   **Plan Mode First**: 코드를 수정하기 전 `SPEC.md`를 작성하여 구현 계획을 제안하고 사용자 승인을 받는다.
*   **Marimo Orchestration**: 안티그래비티 세션 시작 시 `marimo_orchestrator.py start`를 통해 엔진을 가동하고, 종료(Wrap-up) 요청 시 `stop`을 실행하여 자원을 즉시 회수한다.
*   **Manus Method (Scratchpad)**: 진행 상황을 `SCRATCHPAD.md`에 실시간 기록하고 모델의 내부 메모리를 비운다.

## 5. 🚀 EPL Domain Specific Rules (EPL 도메인 특화)
*   **Easy Mode Explanation**: 전술 분석은 축구 초보자도 이해할 수 있도록 쉬운 비유를 섞어 설명한다.
*   **Real-time Reality**: 모든 리포트에 데이터 수집 시각(Timestamp)을 명시한다.
*   **Mobile-First UI**: 모든 시각화 및 UI 요소는 최소 폰트 16px 이상을 유지한다.

## 6. 💰 Activity-based Intent (Lior Alex's Principle)
*   **Behavioral Scoring**: 복합 변수(부상자 등)를 활용한 정밀 시뮬레이션을 고관여(High Intent) 신호로 간주한다.
*   **Contextual Conversion**: 고관여 유저에게는 분석 맥락에 맞는 프리미엄 리포트나 서비스를 제안한다.

## 7. 🧪 Scientific Rigor (K-Dense Inspired)
*   **Sequential Thinking**: 과제를 [데이터 진단 -> 인과관계 추론 -> 최종 검증]의 논리 체인으로 처리한다.
*   **Academic Grounding**: 분석에 Spearman의 Pitch Control 등 검증된 전술 이론을 인용한다.

## 8. 💾 Resource Management (Mac 8GB Optimized)
*   **DuckDB/Polars First**: 대용량 데이터는 Zero-copy 연산을 최우선한다.
*   **Zero-Waste Lifecycle**: 세션 종료 시 `cleanup_agent.py`와 `marimo_orchestrator.py stop`을 호출하여 Mac의 RAM과 디스크를 최적화한다.
*   **No Sprawl (Constraints)**: 파괴적인 명령어 사용 및 데이터 스키마 임의 변경을 금지한다.

## 9. 🧹 Scratchpad Lifecycle (비대화 방지)
*   **Task-level Purge**: 테스크 완료 시 SCRATCHPAD의 상세 로그는 삭제하고 '최종 결과'와 '다음 할 일'만 남긴다.
*   **Smart Compaction**: 파일 크기가 5KB를 초과하면 AI가 즉시 압축(Compaction) 단계를 수행하여 불필요한 로그를 정리한다.

## 10. ♻️ Memory Eviction (자동 소멸 및 지식 증류)
*   **7-Day TTL**: 아카이브 로그(`.agent/scratchpad/archive`)는 생성 7일 후 `cleanup_agent.py`에 의해 자동 소멸된다.
*   **Knowledge Distillation**: 아카이브 삭제 전, 영구적 가치가 있는 지식은 `data/team_memory.json`에 100자 이내로 증류하여 보관한다.
*   **10MB Disk Quota**: `.agent/` 폴더 용량 상한을 10MB로 설정하여 Mac의 저장공간을 보호한다.

## 11. 📱 App-as-a-Tool & Widget-First (Lenny's Principle)
*   **Contextual Discovery**: 사용자가 특정 분석을 요청하면, 관련 '전용 위젯'이나 '에이전틱 미니앱'을 워크플로우 내에서 즉시 제안한다. (예: 이적 루머 언급 시 @TransferWidget 제안)
*   **Chat-Native Widgets**: 정적 텍스트 대신 Marimo, Streamlit, 또는 `generate_image`를 활용한 인터랙티브 시각화 위젯을 우선 제공한다.
*   **MCP Hub Integration**: 외부 도구(GitHub, Slack 등)와의 연결을 위해 MCP를 적극 활용하며, 안티그래비티를 모든 개발/분석 도구의 '배포 허브'로 취급한다.
*   **Fluid Distribution**: 분석 결과나 유용한 워크플로우는 `share_block.py`를 통해 즉시 공유 가능한 형태로 패키징하여 배포한다.

## 12. 🛡️ Agent-Led Data Governance (Jaden's Principle)
*   **Policy-as-Agent**: 보안 및 비즈니스 규정을 에이전트의 페르소나에 내재화하여, 데이터 처리 시 마스킹 및 컴플라이언스를 자동으로 준수하게 한다.
*   **Artifact-Based Traceability**: 가동된 모든 데이터 계보(Lineage)와 처리 근거를 `SCRATCHPAD.md` 및 `COMPLETION_REPORT.md`로 남겨 투명한 감사(Audit) 체계를 구축한다.
*   **Autonomous Quality Guard**: 데이터 가공 후 터미널/브라우저 테스트 도구를 사용하여 정합성 및 품질을 자가 검증(Self-Test)한 후 최종 결과를 사용자에게 보고한다.
*   **Governance by Approval**: 사람은 복잡한 연산 대신 AI가 제시한 '거버넌스 준수 계획'을 승인(Approve)하는 고수준 관리 역할에 집중한다.

## 13. ⚔️ Multi-Model Competition & Collaboration (Lior's Principle)
*   **Ensemble Decision Making**: 단일 모델의 판단에 의존하지 않고, 서로 다른 강점(전술, 데이터, 시장 흐름)을 가진 마이크로 에이전트들이 제안을 내고 경쟁/협업하여 최적의 합의점(Consensus)을 도출한다.
*   **Active Critique**: 에이전트 간 'Red Teaming'을 상시 가동하여, 한 모델이 생성한 코드나 분석 결과를 다른 모델이 비판하고 보완하는 '경쟁적 개선 루프'를 유지한다.
*   **Dynamic Attention**: 사용자의 질문에 따라 가장 적합한 모델/에이전트가 주도권을 가져가는 '주의(Attention) 기반 오케스트레이션'을 수행한다.

## 14. 🎙️ Founder's Wisdom Syndicate (Sasha's Principle)
*   **Sequential Knowledge Building**: Sasha Pavlov가 추천한 20개 팟캐스트(Lenny, All-In, 20VC 등)의 인사이트를 가상의 MCP 레이어로 구축하여, 단순 정보 전달이 아닌 '전략적 컨설팅' 레벨의 답변을 생성한다.
*   **Expert Perspective Blending**: 질문 발생 시 Masters of Scale(확장성), Huberman Lab(효율), Naval(레버리지) 등 다각도로 관점을 투영하여 입체적인 제언을 수행한다.
*   **Actionable Founder Context**: 이론적인 설명에 그치지 않고, 글로벌 창업가들의 실제 실패담과 성공 방정식을 현재 프로젝트 상황(EPL 분석 등)에 맞게 치환하여 제안한다.

## 15. 📉 ML Stack Minimalism (Riad's Principle)
*   **Top 10 Library Strategy**: 산업계 ML 시스템의 90%를 지탱하는 10개 핵심 라이브러리(TensorFlow, PyTorch, Scikit-learn, XGBoost, Transformers, Darts, Prophet, Gensim, PyMC, NumPy/Pandas)를 우선적으로 활용하여 시스템의 복잡도를 낮추고 유지보수성을 극대화한다.
*   **Minimalist Architecture**: 불필요한 라이브러리 도입을 지양하고, 검증된 소수의 도구로 '린(Lean)'한 ML 파이프라인을 구축한다.
*   **Dependency Governance**: 종속성을 최소화하여 보안 취약점을 방지하고, Mac(8GB RAM) 환경에서의 런타임 오버헤드를 줄인다.

## 16. 🧩 Intelligent Chunking Strategy (RAG Optimization)
*   **Adaptive Chunking**: 데이터의 특성에 따라 최적의 청킹 방식을 선택하여 RAG 검색 정확도를 높인다.
    *   **Fixed-Size**: 속도가 중요한 단순 텍스트 처리에 사용 (Overlap 필수).
    *   **Recursive**: 문장/문단 구조 보존이 중요한 일반 문서에 적용.
    *   **Document-Based**: Markdown, JSON 등 구조화된 파일의 계층 구조 유지.
    *   **Semantic**: 주제 변화가 뚜렷하고 고도의 문맥 이해가 필요한 핵심 분석에 사용.
*   **Context Continuity**: 청크 간 중첩(Overlap)을 전략적으로 설정하여 정보의 단절을 방지하고 에이전트의 이해도를 극대화한다.

## 17. 🎨 Sketch-to-Design Workflow (Stitch & Miricle Principle)
*   **Vibe Coding Design**: 사용자님이 종이에 그린 거친 스케치나 낙서를 `@stitch` MCP를 통해 고퀄리티의 UI/광고 배너로 즉시 리디자인(Redesign)한다.
*   **Redesign Orchestration**: [스케치 업로드 -> Stitch 분석 -> 안티그래비티 리디자인 -> 실제 코드/이미지 생성]의 '디자인 프리패스' 워크플로우를 가동한다.
*   **Premium Level Consistency**: 구글 원 AI 프리미엄(AI Pro)의 리소스를 활용하여, 디자인의 구도와 의도는 유지하면서 색상, 폰트, 에셋을 전문가 수준으로 보정한다.

## 18. ⚖️ Anti-Gravity Attention (K-Factor Principle)
*   **Debiasing Attention**: 데이터의 과도한 쏠림(Heavy Hitter)을 방지하기 위해 Candra Alpin의 `K-Factor` 어텐션 원칙을 적용한다. 곱셈(Dot-product) 대신 가산(Add) 중심의 연산을 활용하여 특정 노드로 가중치가 몰리는 '데이터 중력'을 물리적으로 억제한다.
*   **Numerical Stability**: 대규모 데이터셋 처리 시 `Scaled Mode in Logits` 개념을 도입하여 수치적 안정성을 확보하고, 토큰 간의 건강한 경쟁(Global token competition)을 유도한다.
*   **Efficiency Optimization**: 인과적 마스킹(Causal Masking) 오버헤드를 줄이기 위한 K-Factor의 단순화된 연산 구조를 지지하며, Mac(8GB RAM) 환경에서의 학습 및 추론 효율을 극대화한다.

## 19. 🚀 Prompt Caching & Latency Optimization (Efficiency Principle)
*   **Static Context Caching**: 시스템 프롬프트, 구단 기본 정보, 장기 전술 데이터 등 변하지 않는 '고정 컨텍스트'를 우선적으로 캐싱하여 API 비용을 절감하고 응답 속도(Latency)를 극대화한다.
*   **Prompt Architecture**: 캐싱 효율을 높이기 위해 프롬프트 구조를 [Static Instruction -> Permanent Memory -> Dynamic Data] 순으로 배치하여 재사용성을 높인다.
*   **Real-time Feedback**: 캐싱된 컨텍스트를 활용하여 사용자에게 지연 없는(Real-time) AI 피드백을 제공하며, Mac(8GB RAM) 환경에서의 연산 부하를 최소화한다.

## 20. 🏗️ Multimodal Anti-gravity Engine (Naive & Buoyancy Principle)
*   **Naive Thrust (연산의 안티그래비티)**: 텍스트와 이미지 처리 시 무거운 행렬 곱셈($O(N^2)$) 대신 Candra's Naive Formula(가산 기반 선형 결합, $O(N)$)를 사용하여 엔진의 연산 부하를 최소화하고 학습 속도를 극대화한다.
*   **Anti-gravity Lift (신호의 부력 제어)**: 깊은 신경망에서 신호가 아래로 가라앉지(Vanishing) 않도록 `ReZero` 게이팅과 `Jacobian Regularization`을 적용한다. 초기에는 무중력(Zero-Gravity) 상태로 시작하여 신호를 끝까지 전달하고, 점진적으로 필요한 '중력(변환)'을 가한다.
*   **Entropy Repulsion (정보의 균형)**: 이미지 패치와 텍스트 토큰 간의 정보 밀도를 조절하기 위해 엔트로피 척력(Entropy Repulsion Force)을 활용하여, 가중치가 쏠리지 않고 전체적인 맥락을 골고루 띄워 올리게 한다.

## 21. 👔 Executive Reporting Workflow (Claude Skill & Managed Principle)
*   **Skill-Based Automation**: 반복적인 분석 패턴을 `@data-analyst` 스킬로 자산화하여, 데이터 로딩부터 전략 제언까지의 전 과정을 표준화한다.
*   **Executive Output (PPT/PDF)**: 모든 분석의 최종 목적지는 상무님 보고용 PPT(`python-pptx`)와 PDF(`fpdf`)이다. 코드가 아닌 결론(Insight)과 실행 방안(Action Plan) 중심의 시각화 리포트를 자동 생성한다.
*   **Managed Governance**: `managed-settings.json`을 통해 분석에 필요한 라이브러리(Pandas, Seaborn 등)와 보안 권한을 전역적으로 관리하여, 환경에 구애받지 않는 일관된 분석 품질을 유지한다.

## 22. ⚖️ Precision Extraction & Traceability (CUAD Principle)
*   **Precision Clause Extraction**: 수천 페이지의 복잡한 문서(계약서, 규정집)에서CUAD 벤치마크 수준의 고정밀 추출 기법을 적용하여 독소 조항이나 핵심 의무 사항을 자율적으로 식별한다.
*   **Semantic Refactoring**: 코드를 리팩토링하듯 문서를 수정한다. 도메인 지식과 최신 규범을 바탕으로 특정 조항을 정밀하게 타겟팅하여 문맥의 훼손 없이 업데이트한다.
*   **Traceability First**: 에이전트의 모든 판단과 수정 사항은 명확한 '근거 데이터(CUAD 기준, 판례, 공식 규정)'와 연결되어야 하며, 사용자가 승인(Approve)하기 전까지 추론 과정을 투명하게 공개한다.

## 23. ⚡ Ultra-low Latency Visual Generation (TMD Principle)
*   **Step Compression (Distillation)**: 시각적 피드백이나 영상 기반 리포트 생성 시, NVIDIA의 `TMD` 기술처럼 생성 단계를 최소화(1~4단계)하여 초저지연(Real-time) 반응 속도를 확보한다.
*   **Backbone-Flow Head Architecture**: 거시적인 구조(Plan)를 잡는 메인 엔진과 세부 시각 정보를 보정하는 가벼운 플로우 헤드(Flow Head)를 분리하여, Mac(8GB RAM) 환경에서도 고화질 시각 데이터를 빠르게 출력한다.
*   **Interactive Simulation**: 에이전트가 제안하는 시나리오나 분석 결과물(차트, 위젯)을 동적으로 생성할 때, 지연 없는 실시간 렌더링을 통해 사용자 경험의 연속성을 유지한다.

## 24. 🎭 Stitch-Penpot Design Orchestration (Visual Production Principle)
*   **Sketch-to-Prototyping**: 사용자님이 그린 스케치를 `@stitch`가 분석하고, 이를 바탕으로 `@penpot` MCP가 실제 수정 가능한 벡터 디자인 및 프로토타입으로 변환한다.
*   **Design-as-Code Integration**: Penpot의 '코드 기반 디자인' 특성을 활용하여, 디자인에서 실제 React/Tailwind 코드로의 변환 시 오차를 0%에 수렴하게 만든다.
*   **Collaborative Redesign**: 안티그래비티 에이전트가 Penpot 디자인 파일을 직접 수정하거나 에셋을 추출하여, 단순한 그림을 넘어 실제 서비스 가능한 '프러덕션 급 UI'를 자율적으로 완성한다.

## 25. 🌐 Global Insight Auto-Loop (Discovery & Action Principle)
*   **Automated Harvesting**: Chip Huyen, Karpathy, Lenny's Newsletter 등 전 세계 기술 리더들의 최신 인사이트를 매일 자동으로 수집하여 아침 보고서(`.agent/insights/reports/`)를 생성한다.
*   **Instruction-Driven Synthesis**: 사용자님이 보고서 내용 중 특정 기술이나 전략을 적용하도록 지시하면, 안티그래비티 에이전트는 즉시 관련 규칙(`rules.md`)과 에이전트 가이드(`agents.md`)를 업데이트하고 필요 코드를 수정한다.
*   **Resource-Aware Cleanup**: Mac(8GB RAM)의 자원을 보호하기 위해, 읽은 원본 데이터와 1주일 이상 지난 리포트는 자동으로 삭제(Delete-on-Complete)하여 시스템 부하를 최소화한다.

## 26. 🧠 2025 Applied AI Frontier (State of AI 2025 Principle)
*   **System 2 Reasoning**: 단순 반응형 답변을 넘어 `o1`, `DeepSeek R1`과 같은 추론 모델을 활성화하여 복잡한 코딩 및 설계 문제에 대해 '생각하는 단계(Chain-of-Thought)'를 명시적으로 거친다.
*   **T-Shaped Agent Specialization**: 모든 에이트가 범용일 필요는 없다. 특정 도메인(법률, 데이터, 시각화)에 깊은 전문성을 가진 에이전트들이 MCP 표준을 통해 유기적으로 통합되는 'T자형 에이전트 연합체'를 구축한다.
*   **Trajectory-Based Evaluation**: 최종 결과물만 평가하는 것이 아니라, 에이전트가 문제를 해결하기 위해 거친 일련의 단계(Trajectory)를 추적하고 검증한다. 실패 시 트래젝토리를 분석하여 '자기 교정(Self-Correction)' 루프를 가동한다.
*   **Context Engineering & Lost-in-the-Middle Defense**: 단순 프롬프트를 넘어 컨텍스트 윈도우 내의 정보 배치와 검색 밀도를 알고리즘적으로 최적화한다. 정보가 중간에서 누락되지 않도록 Agentic Retrieval 및 그래프 기반 청킹을 활용한다.

## 27. 🧠 Agentic Memory Management (AgeMem Principle)
*   **Autonomous Resource Management**: 에이전트가 자신의 메모리 수명 주기를 직접 관리한다. 6가지 핵심 도구(`ADD`, `UPDATE`, `DELETE`, `RETRIEVE`, `SUMMARY`, `FILTER`)를 활용하여 컨텍스트 예산 내에서 최적의 정보 밀도를 유지한다.
*   **Memory Lifecycle Strategy**:
    *   `ADD/UPDATE`: 중요한 대화나 관찰 결과를 메타데이터와 함께 장기 기억(LTM)으로 자산화한다.
    *   `SUMMARY/FILTER`: 8GB RAM 환경 보호를 위해 불필요한 노이즈를 즉각 제거하고 긴 대화를 압축한다.
    *   `RETRIEVE`: 필요한 순간에만 장기 기억에서 핵심 정보를 인출하여 컨텍스트 지지력을 확보한다.
*   **Step-wise Reflection**: 모든 작업 완료 후, 저장된 기억이 문제 해결에 기여했는지를 스스로 평가하고 그 궤적(Trajectory)을 기록하여 다음 분석의 전략을 고도화한다.

## 28. 🧬 Advanced Transformer Architecture (Gating & Log Principle)
*   **Architectural Synergy (GAU & MQA)**: 시계열 및 텍스트 데이터 처리 시, NLP 베스트 프랙티스인 `Gating Attention Units (GAU)`와 `Multi Query Attention (MQA)`을 우선적으로 검토한다. 이는 수렴 속도와 추론 효율을 동시에 잡는 핵심 전략이다. 
*   **Numerical Boosters (SwiGLU & Rezero)**: 신경망의 활성화 함수로 `SwiGLU`를, 잔차 연결의 안정성을 위해 `Rezero` 게이팅을 적용하여 8GB RAM 환경에서도 깊고 안정적인 학습을 유도한다.
*   **Convergence Monitoring**: 학습 중 그래디언트 소실이나 수렴 정체 발생 시, 어텐션 메커니즘을 `Multi-Head Attention`으로 회항하거나 어텐션 맵의 엔트로피를 체크하여 즉시 교정한다.

## 29. 📜 Professional Research Logging (Candra's Log Principle)
*   **Evidence-Based Traceability**: 모든 모델 학습 및 실험 시 반드시 `RESEARCH_LOG.md`에 [날짜 - 실험 내용 - 결과(Loss/Metric) - 인사이트(성공/실패 원인)]를 기록한다. 
*   **Failure as Knowledge**: 단순 결과 나열이 아닌, "왜 이 특정 레이어(예: MQA)가 시계열 데이터에서 더 나은 성능을 보였는지"에 대한 기술적 논평을 포함한다.
*   **Visual Evidence**: 학습 곡선(Loss Curve)이나 어텐션 히트맵 등 시각적 증거를 함께 관리하여 분석의 신뢰도를 높인다.

---
*Applied: Hackathon Tips, Lior Alex's Intent/Competition, K-Dense Scientific, GeekNews Spec, Manus Method, Marimo Lifecycle, Lenny's Strategy, Jaden's Governance, Sasha's Founder Wisdom, Riad's ML Minimalism, Simone's Chunking, Google Stitch Design, Candra's Attention/Naive Engine, Prompt Caching, Claude Executive Skills, CUAD Precision, NVIDIA TMD Real-time, Stitch-Penpot Orchestration, Global Insight Loop, State of Applied AI 2025, AgeMem Agentic Memory, GAU/MQA/SwiGLU Architecture (2026.01.21)*
