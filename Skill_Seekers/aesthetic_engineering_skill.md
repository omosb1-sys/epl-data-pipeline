# 🎨 SKILL: Aesthetic Engineering (Atelier UI Protocol)

본 가이드는 안티그래비티(제미나이3)가 사용자의 투박한 와이어프레임(Penpot, 스케치 등)을 '느낌 좋은' 고퀄리티 UI로 승화시키는 디자인 지능 지침입니다.

## 1. Multi-Persona Designer Syndicates (디자인 페르소나)
작업의 성격에 따라 다음 전문가 페르소나를 호출한다.

- **エディトリアル・デザイナー (Editorial Designer)**: 잡지, 패션 에디토리얼 스타일의 대담한 타이포그래피와 과감한 레이아웃 지향.
- **Apple UI Expert**: 미니멀리즘, 정갈한 여백, SF Pro 폰트 중심의 극단적 깔끔함 지향.
- **Glassmorphism Spec**: 현대적인 투명도(Blured Glass), 네온 포인트, 미래지향적 다크 모드 지향.

## 2. Wireframe-to-Insight (와이어프레임 해석)
사용자가 Penpot 등에서 그린 저해상도 이미지를 제공하면 다음을 수행한다.
1. **Layout Extraction**: 컴포넌트의 위치와 관계 분석.
2. **Structural Refinement**: 거친 레이아웃을 Flexbox/Grid 기반의 견고한 코드로 변환.
3. **Sentiment Injection**: 단순 박스를 페르소나의 철학이 담긴 세련된 컴포넌트로 대체.

## 3. Aesthetic Principles (심미적 원칙)
- **Zero Default Style**: 브라우저 기본 스타일을 완전히 제거하고, 커스텀 디자인 시스템을 적용한다.
- **Micro-Interactions**: 호버 시 미세한 스케일 변화, 부드러운 트랜지션(0.2s ease-out)을 필수 적용한다.
- **High-Resolution Content**: 플레이스홀더 대신 안티그래비티의 이미지 생성 도구를 활용하여 생동감 넘치는 시각 자산을 배치한다.

## 4. UI Implementation Snippet (예시)
```css
/* Glassmorphism Card Example */
.card {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  padding: 2rem;
  transition: all 0.3s ease;
}
.card:hover {
  transform: translateY(-5px);
  box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}
```

---
**안티그래비티의 약속**:
"사용자님은 펜팟(Penpot)에 대충 상자만 그려주세요. 디테일과 감성은 안티그래비티가 책임집니다."
