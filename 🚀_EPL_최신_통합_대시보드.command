#!/bin/bash
# 현재 스크립트의 절대 경로 확보
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$BASE_DIR"

echo "📂 작업 디렉토리: $BASE_DIR"

# 가상환경 활성화 (EPL 전용 환경 선순위)
if [ -d ".venv_epl" ]; then
    source .venv_epl/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# 필수 패키지 확인 및 조용한 설치
pip install -q watchdog

echo "🚀 EPL TOON 통합 대시보드를 실행합니다..."
python3 -m streamlit run epl_project/epl_toon_dashboard.py \
    --server.runOnSave=true \
    --server.fileWatcherType=watchdog \
    --browser.gatherUsageStats=false

# 오류 발생 시 창이 닫히지 않게 대기
if [ $? -ne 0 ]; then
    echo "❌ 실행 중 오류가 발생했습니다. 위 메시지를 확인하세요."
    read -p "엔터 키를 누르면 종료됩니다..."
fi
