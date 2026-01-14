#!/bin/bash
# 이 파일이 있는 경로로 이동
cd "$(dirname "$0")"

# 가상환경 활성화
source .venv/bin/activate

# Streamlit 앱 실행 및 브라우저 자동 호출
# --server.headless false 옵션은 브라우저를 강제로 띄우는 역할을 합니다.
python -m streamlit run src/app_unified_analyst.py --server.headless false
