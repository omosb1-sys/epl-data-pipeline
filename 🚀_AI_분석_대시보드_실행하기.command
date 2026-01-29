#!/bin/bash

# K-리그 AI 분석 대시보드 실행 스크립트
# GEMINI.md Protocol 준수

echo "🚀 K-리그 AI 분석 대시보드 시작..."
echo ""

# 환경변수 확인
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  경고: GEMINI_API_KEY 환경변수가 설정되지 않았습니다."
    echo ""
    echo "다음 명령어로 API 키를 설정하세요:"
    echo "export GEMINI_API_KEY='your-api-key-here'"
    echo ""
    echo "API 키 발급: https://makersuite.google.com/app/apikey"
    echo ""
    echo "기본 통계 분석은 API 키 없이도 사용 가능합니다."
    echo ""
fi

# 프로젝트 루트로 이동
cd "$(dirname "$0")"

# Streamlit 실행
echo "📊 Streamlit 서버 시작 중..."
streamlit run src/app_kleague_ai.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false

echo ""
echo "✅ 대시보드가 종료되었습니다."
