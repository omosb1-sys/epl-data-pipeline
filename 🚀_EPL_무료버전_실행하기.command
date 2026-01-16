#!/bin/bash
# 0-Cost Serverless EPL Manager 실행 스크립트 (가상환경 자동 설치 버전)

# 현 스크립트 파일의 위치로 이동
cd "$(dirname "$0")"

# UV path
UV_BIN="/Users/sebokoh/.local/bin/uv"

echo "🚀 [EPL-X Lite] 서버리스 모드로 시작합니다..."

# 가상환경 생성 (없을 경우)
if [ ! -d ".venv_epl" ]; then
    echo "📦 가상환경 생성 중..."
    $UV_BIN venv .venv_epl --python 3.12
fi

# 필수 부품 체크 및 설치 (이미 있으면 빠르게 스킵됨)
echo "🛠️ 필수 부품 체크 및 업데이트 중..."
$UV_BIN pip install --python .venv_epl/bin/python beautifulsoup4 requests lxml streamlit pandas torch scikit-learn joblib xgboost lightgbm statsmodels plotly

echo "📂 Project Root: $(pwd)/epl_project"

# 실행
.venv_epl/bin/streamlit run epl_project/app.py --server.port 8503
