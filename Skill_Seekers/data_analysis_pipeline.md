# 📊 데이터 분석 전체 파이프라인

## 1. 문제 정의 및 목표 설정
- 해결하고자 하는 비즈니스 문제 명확화
- 분석의 성공 기준(KPI) 정의
- 예상 산출물 및 활용 방안 계획

## 2. 데이터 수집
- 데이터 소스 파악 (DB, API, 파일, 웹 크롤링 등)
- 필요한 데이터 범위 및 기간 결정
- 데이터 수집 및 저장

## 3. 탐색적 데이터 분석(EDA)
- 데이터 구조 파악 (shape, columns, dtypes)
- 기술 통계량 확인 (mean, median, std 등)
- 데이터 분포 시각화
- 변수 간 상관관계 분석
- 패턴 및 인사이트 발견

## 4. 데이터 전처리
- 결측치 처리 (제거, 대체, 보간)
- 이상치 탐지 및 처리
- 데이터 타입 변환
- 중복 데이터 제거
- 데이터 정규화/표준화

## 5. 피처 엔지니어링
- 새로운 변수 생성 (파생 변수)
- 범주형 변수 인코딩
- 변수 선택 및 차원 축소
- 시계열 특성 추출 (필요시)

## 6. 모델 선택 및 학습
- 문제 유형에 맞는 알고리즘 선택
  - 분류: Logistic Regression, Random Forest, XGBoost 등
  - 회귀: Linear Regression, Ridge, Lasso 등
  - 군집: K-Means, DBSCAN 등
- 훈련/검증/테스트 데이터 분할
- 모델 훈련 및 교차 검증

## 7. 모델 평가 및 최적화
- 성능 지표 계산 (Accuracy, F1, RMSE 등)
- 혼동 행렬 분석
- 하이퍼파라미터 튜닝 (Grid Search, Random Search)
- 모델 앙상블 고려

## 8. 결과 시각화 및 인사이트 도출
- 주요 발견사항 정리
- 효과적인 차트 및 그래프 작성
- 비즈니스 관점에서의 해석

## 9. 리포트 작성 및 결과 공유
- 분석 과정 및 결과 문서화
- Jupyter Notebook 정리
- 대시보드 생성 (필요시)

---

### 💡 추천 도구 및 라이브러리
- **데이터 처리**: pandas, numpy
- **시각화**: matplotlib, seaborn, plotly
- **머신러닝**: scikit-learn, xgboost, lightgbm
- **딥러닝**: tensorflow, pytorch (필요시)
- **통계 분석**: scipy, statsmodels
