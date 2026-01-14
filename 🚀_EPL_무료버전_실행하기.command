#!/bin/bash
# 0-Cost Serverless EPL Manager 실행 스크립트

# Python Path (User's environment)
UV_PATH="/Users/sebokoh/Library/Python/3.9/bin/uv"

echo "🚀 [EPL-X Lite] 서버리스 모드로 시작합니다..."
echo "📂 Project Root: $(pwd)/epl_project"

# 실행 (Port 8503 사용)
$UV_PATH run --python 3.12 streamlit run epl_project/app.py --server.port 8503
