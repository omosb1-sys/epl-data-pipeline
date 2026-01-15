#!/bin/bash
# 0-Cost Serverless EPL Manager 실행 스크립트 (경로 문제 수정 버전)

# 현 스크립트 파일의 위치로 이동
cd "$(dirname "$0")"

# Python Path (User's environment)
UV_PATH="/Users/sebokoh/Library/Python/3.9/bin/uv"

echo "🚀 [EPL-X Lite] 서버리스 모드로 시작합니다..."
echo "📂 Project Root: $(pwd)/epl_project"

# 직접 app.py 존재 여부 확인
if [ ! -f "epl_project/app.py" ]; then
    echo "❌ 에러: epl_project/app.py 파일을 찾을 수 없습니다."
    echo "현재 위치: $(pwd)"
    ls -R epl_project
    exit 1
fi

# 실행 (Port 8503 사용)
$UV_PATH run --python 3.12 streamlit run epl_project/app.py --server.port 8503
