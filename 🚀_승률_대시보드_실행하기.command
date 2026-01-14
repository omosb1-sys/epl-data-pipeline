#!/bin/bash
# 이 파일이 있는 경로로 이동
cd "$(dirname "$0")"

# 가상환경 활성화
source .venv/bin/activate

echo "🚀 K-League 승률 분석 대시보드를 실행합니다..."
# Streamlit 앱 실행 (포트 8503 사용 - 다른 앱과 충돌 방지)
python -m streamlit run src/app_win_rate_dashboard.py --server.port 8503 --server.headless false
