# 📊 SKILL: Advanced Data Science Toolkit

본 가이드는 제미나이3(Antigravity)가 데이터 분석 코드를 작성할 때 참조하는 고급 기술 스택 지침입니다.

## 1. 📈 통계 분석 (Statistics)
- **Statsmodels**: 회귀 분석(OLS), 시계열 분해, 잔차 분석에 사용.
- **Pingouin**: T-test, ANOVA, 상관관계 분석을 Pandas 데이터프레임 형식으로 직관적으로 수행.

## 2. 🤖 머신러닝 & 딥러닝 (ML & DL)
- **XGBoost / LightGBM**: 정형 데이터의 분류 및 회귀 예측 시 기본 모델로 채택.
- **PyTorch**: 딥러닝 모델 설계 및 복잡한 텐서 연산 시 사용.

## 3. ⏳ 시계열 분석 (Time Series)
- **Prophet**: 비즈니스 주기성이 강한 시계열 예측에 우선 적용.
- **pmdarima**: ARIMA 모델의 자동 하이퍼파라미터 최적화(auto_arima)에 활용.

## 4. ⚡ 빅데이터 & 최적화 (Big Data)
- **Polars**: 대용량 CSV 로딩 및 고속 데이터 처리가 필요할 때 Pandas 대신 사용.
- **Dask**: 병렬 처리가 필요한 연산 집약적 작업에 적용.

## 5. 🎨 인터랙티브 시각화 (Visualization)
- **Plotly**: HTML 기반의 대화형 그래프, 줌/인터랙션이 필요한 대시보드 제작 시 사용.
- **Altair**: 선언적 통계 시각화가 필요할 때 활용.

---
**제미나이3의 적용 원칙**:
- 위 라이브러리가 설치되어 있음을 인지하고, 분석 상황에 따라 가장 효율적인 라이브러리를 제안하고 코드에 반영한다.
- 특히 맥(Mac) 환경에서 `AppleGothic` 폰트와 호환되도록 시각화 설정을 자동화한다.
