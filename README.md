# ⚽ EPL-X: AI Football Manager (2025-26)

![EPL-X Logo](https://img.shields.io/badge/EPL_X-AI_Analytics-blue?style=for-the-badge&logo=premier-league)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

> **"축구는 데이터다."**
> 빅데이터와 딥러닝(Deep Learning)을 결합하여 프리미어리그(EPL) 경기 결과, 이적 시장, 팀 전력을 정밀 분석하는 AI 대시보드입니다.

---

## 🚀 주요 기능 (Key Features)

### 1. 📊 실시간 대시보드
- 구단별 **AI 전력 지수(Power Index)** 산출
- 감독, 구단 가치, 최근 경기력 등 핵심 KPI 시각화
- 2025-26 시즌 최신 스쿼드 및 전술 데이터 반영

### 2. 🧠 AI 승부 예측 시뮬레이터
- **Deep Learning (PyTorch)** + **RandomForest** 앙상블 모델 탑재
- 홈/원정 어드밴티지, 부상자 현황, 팀 분위기(Mood) 변수 조작 가능
- **SHAP 분석** 기반의 "왜 이 팀이 이길까?" 인과관계 리포트 제공

### 3. 🔁 이적 시장 통합 센터
- **Google News & BBC** 실시간 크롤링을 통한 이적 루머 감지
- '파브리지오 로마노(Romano)' 등 공신력 높은 소스 필터링
- AI가 분석한 선수 영입/방출 확률(%) 시각화

### 4. 📰 뉴스 & 인사이트
- 전 구단 관련 최신 뉴스 자동 수집 및 요약
- 긍정/부정(Sentiment) 분석을 통한 구단 분위기 진단

---

## 💾 설치 및 실행 (Installation)

이 프로젝트는 **Serverless** 환경에서도 동작하도록 설계되었습니다.

### 로컬 실행 (Mac/Windows)
```bash
# 1. 저장소 복제
git clone https://github.com/your-id/epl-x-manager.git
cd epl-x-manager

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 앱 실행
streamlit run epl_project/app.py
```

### ☁️ 클라우드 배포 (Streamlit Cloud)
이 리포지토리를 [Streamlit Cloud](https://streamlit.io/cloud)에 연결하기만 하면 즉시 웹 앱으로 배포됩니다.
- **Entry point**: `epl_project/app.py`
- **Python version**: 3.9 or 3.10

---

## 🛠 기술 스택 (Tech Stack)
- **Frontend**: Streamlit
- **Backend/AI**: PyTorch, Scikit-learn, XGBoost
- **Data Collection**: BeautifulSoup4, Requests (Google News, BBC)
- **Deployment**: Docker Support (Optional)

---

## 📝 License
This project is for educational and analytical purposes. Data provided by API-Football and various open sources.

---
*Created by Senior Analyst Antigravity*
