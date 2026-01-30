
import streamlit as st
import os
import sys

# [GATEWAY] Antigravity Global Entry Point
# 리포지토리 루트에서 epl_project/app.py를 실행하기 위한 경로 조정 및 래핑

# 1. 경로 추가: epl_project 내부 모듈들을 인식할 수 있도록 함
project_root = os.path.join(os.path.dirname(__file__), "epl_project")
sys.path.append(project_root)

# 2. 작업 디렉토리 변경 (Data/Assets 경로 호환성 확보)
os.chdir(project_root)

# 3. 실제 서비스 로직 실행
try:
    with open("app.py", "r", encoding="utf-8") as f:
        code = f.read()
    exec(code, globals())
except Exception as e:
    st.error(f"🚀 배포 서버 구동 오류: {e}")
    st.info("현재 디렉토리: " + os.getcwd())
    st.write("사용 가능한 파일들:", os.listdir())
