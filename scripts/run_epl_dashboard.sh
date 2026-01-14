#!/bin/bash
# EPL-X Manager 대시보드를 실행하는 단축 스크립트입니다.
echo "🚀 EPL-X Manager Dashboard를 실행합니다..."

# uv를 사용하여 필요한 라이브러리와 함께 스트림릿 실행
/Users/sebokoh/Library/Python/3.9/bin/uv run --python 3.12 \
    --with streamlit \
    --with mysql-connector-python \
    --with pandas \
    --with watchdog \
    --with beautifulsoup4 \
    --with requests \
    streamlit run "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/app.py" --server.port 8505
