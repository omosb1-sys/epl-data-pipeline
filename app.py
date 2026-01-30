
import streamlit as st
import os
import sys

# [CRITICAL FIX] Antigravity Global Entry Point v2.0
# Streamlit Cloud 환경에서 서브 폴더(epl_project) 모듈 인식을 위한 경로 강제 주입

# 1. 절대 경로 계산 인터페이스
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "epl_project")

# 2. PYTHONPATH 최상단에 프로젝트 폴더 주입 (모든 import 경로 해결)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, current_dir)

# 3. 환경 변수 설정 (데이터 로딩용)
os.environ["PROJECT_ROOT"] = project_root

# 4. 실제 앱 실행 (모듈 임포트 방식)
try:
    # epl_project/app.py 내용을 직접 실행하여 Streamlit context 유지
    app_path = os.path.join(project_root, "app.py")
    with open(app_path, "r", encoding="utf-8") as f:
        code = f.read()
    
    # app.py 내부에서 'epl_project/' 경로를 사용하는 부분들을 보정하기 위해 코드 실행 전 cwd 변경
    os.chdir(project_root)
    
    exec(code, globals())
    
except Exception as e:
    st.error("🚀 **Antigravity 배포 엔진 치명적 오류**")
    st.exception(e)
    st.info(f"검색된 경로: {sys.path[:3]}")
    if os.path.exists(project_root):
        st.success("✅ epl_project 폴더 감지됨")
        st.write("내부 파일:", os.listdir(project_root))
    else:
        st.error("❌ epl_project 폴더를 찾을 수 없습니다.")
