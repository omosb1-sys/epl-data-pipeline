
# EPL Project Specific Rules

이 설정 파일은 **EPL 데이터 분석 프로젝트(`epl_project`)**에만 적용되는 로컬 규칙입니다.

## 1. 🏗️ Tech Stack
- **Dashboard**: Streamlit (Python)
- **Data Processing**: Pandas, DuckDB (Polars는 대용량 처리 시 보조 사용)
- **AI/ML**: PyTorch (Deep Learning), SHAP (XAI)
- **Visualization**: Plotly, Altair (Interactive Graphs)
- **Styling**: Custom CSS with Glassmorphism & Dark Mode

## 2. 🧪 Project-Specific Workflows
- **데이터 업데이트**: `collect_advanced_data.py` 실행 시 반드시 최신 팀 데이터(`team_match_results.csv`)를 먼저 백업할 것.
- **모델 학습**: 새로운 예측 모델 실험은 `internal/experiment_engine.py`를 통해 버전 관리할 것.
- **배포**: Streamlit Cloud 배포 전 `pip freeze > requirements.txt`로 의존성을 고정할 것.

## 3. 🚨 Data Privacy
- **Api-Football Key**: `RAPIDAPI_KEY`는 절대 코드에 하드코딩하지 말고 환경 변수나 `st.secrets`로 관리할 것.
