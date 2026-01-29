---
name: premium-ux-standards
description: Vercel 및 고급 웹 디자인 표준을 통합한 프리미엄 UI/UX 가이드라인
user-invocable: true
---

# 💎 Premium UX Standards: Vercel 기반의 초격차 인터페이스 지침

본 스킬은 Vercel Labs의 Agent Skills와 Web Interface Guidelines를 집대성하여, 안티그래비티가 생성하는 모든 웹 제품이 최상급 품질을 갖추도록 강제합니다.

## ⚡ 1. 성능 및 엔지니어링 (Vercel Standards)
1.  **Waterfalls Elimination**: 독립적인 데이터 요청은 `Promise.all()`을 사용하고, `Suspense`를 활용한 스트리밍 렌더링을 지향합니다.
2.  **Bundle Optimization**: 배럴 파일(`.index.ts`) 사용을 금지하고, 컴포넌트 단위 지연 로딩(dynamic import)을 기본으로 합니다.
3.  **State Persistence**: 필터, 정렬 등 주요 UI 상태는 URL Query Parameter와 상시 동기화합니다.

## 🎨 2. 프리미엄 디자인 및 감각 (Premium Aesthetics)
1.  **Typography**: 수치 데이터에는 `tabular-nums`를 적용하고, 헤드라인에는 `text-wrap: balance`를 사용하여 가독성을 극대화합니다.
2.  **Micro-interactions**: 버튼 호버, 로딩 전환 등 모든 인터랙션에 부드러운 트랜지션과 시각적 피드백을 적용합니다. (Glassmorphism, Gradients 적극 활용)
3.  **Visual Hierarchy**: 핵심 행동(CTA) 버튼은 주변 요소보다 채도가 높은 색상을 사용하여 사용자의 시선을 즉각적으로 유도합니다. (Rule 7.5 연계)

## ♿ 3. 접근성 및 안전성 (A11y & Safety)
1.  **Semantic Elements**: 반드시 의미에 맞는 HTML 태그를 사용합니다 (`<button>` vs `<a>`). `<div onClick>`은 엄격히 금지합니다.
2.  **Interactive Safety**: 삭제와 같은 파괴적 행동 시 반드시 확인 모달을 띄우거나 Undo 기능을 제공합니다.
3.  **Aria-Labeling**: 모든 입력 폼과 아이콘 버튼에 접근성을 위한 `aria-label`을 필수적으로 부여합니다.

---
*Unified by Antigravity (Premium UX & Design Lab) - 2026.01.26*
