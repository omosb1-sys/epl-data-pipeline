# 🤖 2026년 데이터 분석가를 위한 AI 도구 가이드

민수님과의 대화에서 언급된 9가지 핵심 도구와, 파이썬 개발자인 당신을 위한 **"설치형 대안 도구"**를 정리해 드립니다.

## 1. 🌐 웹 기반 핵심 도구 (대화 내용 기반)

이 도구들은 대부분 웹 브라우저(SaaS)에서 실행되며, 별도의 설치가 필요 없습니다. 즐겨찾기에 등록해두고 사용하세요.

| 분야 | 도구명 | 설명 | URL (Example) |
| :--- | :--- | :--- | :--- |
| **분석(Excel)** | **Julius AI** | "연도별 매출 요약해줘" 텍스트로 분석 | [julius.ai](https://julius.ai) |
| **분석(Spreadsheet)** | **Quadratic AI** | 엑셀 + Python/SQL이 합쳐진 형태 | [quadratichq.com](https://quadratichq.com) |
| **시각화** | **Bricks** | 텍스트 입력으로 대시보드/차트 생성 | [bricks.do](https://bricks.do) |
| **엔터프라이즈** | **Zebra BI** | 전문적인 재무/비즈니스 대시보드 (Excel Add-in) | [zebrabi.com](https://zebrabi.com) |
| **PPT 제작** | **Gamma AI** | 텍스트로 PPT 슬라이드 자동 생성 | [gamma.app](https://gamma.app) |
| **인포그래픽** | **Piktochart AI** | 타임라인, 인포그래픽 자동 생성 | [piktochart.com](https://piktochart.com/ai) |
| **이미지** | **Ideogram** | 텍스트가 포함된 고화질 이미지 생성 | [ideogram.ai](https://ideogram.ai) |
| **영상** | **Synthesia AI** | 아바타가 설명하는 영상 제작 | [synthesia.io](https://synthesia.io) |
| **웹사이트** | **Lovable** | "웹사이트 만들어줘"로 앱/웹 구축 | [lovable.dev](https://lovable.dev) |

---

## 2. 🐍 파이썬 개발자를 위한 "설치형" 대안 (추천)

당신은 코드를 다룰 줄 아는 분석가이므로, 웹 서비스의 한계를 넘어 **로컬 환경(VS Code, Streamlit)**에서 직접 쓸 수 있는 강력한 라이브러리를 설치해 드렸습니다.

### 🌟 대체 도구 1: [PyGWalker] (Bricks의 완벽한 대안)
- **개요**: "Bricks"처럼 드래그 앤 드롭으로 데이터를 탐색하고 시각화할 수 있는 Tableau 대체재입니다.
- **설치**: `pip install pygwalker` (이미 설치 완료)
- **장점**: 데이터를 외부 서버로 보내지 않고 로컬에서 안전하게 분석 가능. Streamlit과 연동됨.
- **사용법**:
  ```python
  import pandas as pd
  import pygwalker as pyg
  
  df = pd.read_csv("data.csv")
  pyg.walk(df) # Jupyter Notebook에서 UI 실행
  ```

### 🌟 대체 도구 2: [PandasAI] (Julius의 대안)
- **개요**: DataFrame에 대고 "매출이 가장 높은 달은 언제야?"라고 물어볼 수 있는 라이브러리입니다.
- **설치**: `pip install pandasai`
- **사용법**:
  ```python
  from pandasai import SmartDataframe
  sdf = SmartDataframe(df)
  sdf.chat("Top 5 팀 시각화해줘")
  ```

---

## 3. 🚀 바로 실행해보기 (PyGWalker 데모)

`src/demo_pygwalker.py` 파일을 실행하거나 Streamlit 앱에 통합하여 "Bricks" 같은 경험을 바로 느껴보세요.
