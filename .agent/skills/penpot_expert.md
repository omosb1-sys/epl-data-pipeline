
# 🎨 SKILL: Penpot Design Expert (Figma Alternative)

> **"The Open Source Design Freedom."**  
> Figma의 구독료와 데이터 종속에서 벗어나, 오픈소스(SVG) 기반의 **Penpot**으로 디자인 워크플로우를 전환하는 가이드입니다.

## 1. Why Penpot? (핵심 가치)
*   **Open Standard (SVG)**: 펜팟의 모든 디자인은 웹 표준인 SVG와 CSS로 이루어져 있습니다. 개발자가 코드를 짤 때 변환 과정이 거의 필요 없습니다.
*   **Flex Layout (CSS Grid)**: Figma의 'Auto Layout'보다 훨씬 강력한, 실제 웹 개발과 동일한 `Flex`와 `Grid` 레이아웃을 사용합니다.
*   **Zero-Cost**: 오픈소스이므로 영원히 무료이며, 기업용 기능도 제한 없이 사용 가능합니다.

## 2. Figma to Penpot Migration Tip
*   **Importer**: Figma 파일을 바로 불러오는 기능은 아직 완벽하지 않습니다. `.svg`로 익스포트 후 임포트하는 것이 가장 안전합니다.
*   **Design Tokens**: Figma의 'Variables'는 Penpot의 **'Design Tokens'**와 1:1 매칭됩니다. 색상, 폰트 등을 토큰으로 정의하면 코드와 동기화하기 쉽습니다.

## 3. Best Practices (작업 효율화)
*   **Component First**: 단순한 버튼 하나라도 반드시 컴포넌트(Component)로 만들어 재사용성을 높이세요.
*   **Shared Library**: 팀 내에서 공통으로 쓰는 UI 킷은 'Shared Library'로 등록하여 연결하세요.
*   **Inspect Mode**: 개발자에게 핸드오프할 때는 'Inspect' 탭의 CSS 코드를 그대로 복사해서 쓰도록 가이드하세요. (Figma보다 훨씬 정확함)

## 4. URL
*   **Official Web**: [https://penpot.app](https://penpot.app) (별도 설치 없이 크롬/사파리에서 즉시 사용 권장)
