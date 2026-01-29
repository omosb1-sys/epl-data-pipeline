# 📉 Antigravity ML Stack Map (Riad's Principle)

이 문서는 Riad Anas가 제시한 "90%의 ML 시스템을 지탱하는 10대 핵심 라이브러리"를 안티그래비티 프로젝트에 최적화하여 정리한 지식 베이스입니다. 프로젝트 설계 시 이 맵을 참조하여 기술적 부채를 최소화하고 효율을 극대화합니다.

## 🧭 ML 라이브러리 치트 시트 (The Top 10)

### 🧱 Core Production Stack (사용자 숙련 도구)
> "빠르고 안정적인 분석을 위한 사용자님의 주력 무기입니다."

| 라이브러리 | 주 용도 | 난이도 | Antigravity 활용 포인트 |
| :--- | :--- | :--- | :--- |
| **Scikit-learn** | 머신러닝 베이스라인 | 중간 | 모델링의 기초 및 검증 |
| **XGBoost** | 고성능 정형 데이터 분석 | 중간 | 경기 결과 예측의 핵심 엔진 |
| **NumPy/Pandas** | 데이터 핸들링 | 낮-중 | 가장 많이 쓰이는 데이터 준비 도구 |
| **PyTorch** | 유연한 딥러닝 구현 | 중-상 | `ai_loader.py` 추론 엔진 |

### 🚀 Strategic Expansion Stack (새롭게 도입할 도구)
> "심화 분석이 필요할 때 안티그래비티가 가이드를 제공하며 함께 익혀나갈 도구입니다."

| 라이브러리 | 주 용도 | 난이도 | Antigravity 활용 포인트 |
| :--- | :--- | :--- | :--- |
| **Darts** | 시계열 특화 예측 | 중간 | 승점/득점 추세 정밀 예측 |
| **Gensim** | 텍스트 주제 추출 | 중간 | 대량 뉴스 키워드/토픽 분석 |
| **PyMC** | 확률적 신뢰도 분석 | 중간 | 예측 결과의 불확실성 시각화 |
| **TensorFlow** | 대규모 신경망 구축 | 높음 | 확장성 있는 딥러닝 아키텍처 |
| **Transformers** | 최신 AI/LLM 활용 | 중-상 | 뉴스 감성 분석 및 리포트 자동화 |

---

## 🛠️ 신규 도입 5대 도구 상세 활용법

### 1. 🎯 Darts (Time-Series)
- **언제 써야 하나?**: "다음 5경기 토트넘의 득점 확률은?"
- **한 줄 문법**: `model = ExponentialSmoothing(); model.fit(train); forecast = model.predict(5)`

### 2. 🧠 Gensim (NLP Topics)
- **언제 써야 하나?**: "오늘 쏟아진 100건의 기사 중 가장 중요한 이적 이슈는?"
- **한 줄 문법**: `LdaModel(corpus, num_topics=3)`로 숨겨진 주제 추출

### 3. ⚖️ PyMC (Bayesian)
- **언제 써야 하나?**: "손흥민 선발 시 승률 70%라는 데이터가 얼마나 정확한가?"
- **한 줄 문법**: `pm.sample()`을 통해 확률 분포상의 신뢰 구간 확보

### 4. 🚀 TensorFlow (Deep Learning)
- **언제 써야 하나?**: "이미지나 대규모 데이터 셋을 학습한 복잡한 모델링이 필요할 때"
- **한 줄 문법**: `tf.keras.Sequential`로 층을 쌓아 직관적으로 모델링

### 5. 🤖 Transformers (Hugging Face)
- **언제 써야 하나?**: "전문가처럼 뉴스 리포트를 요약하거나 감정을 읽어낼 때"
- **한 줄 문법**: `pipeline("sentiment-analysis")` 호출만으로 AI 분석기 가동

---
*Created by Antigravity (Riad's ML Minimalism) - 2026.01.21*
