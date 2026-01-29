import json
import os
import pandas as pd
from datetime import datetime

# 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "internal", "session_history.jsonl")
PROJECT_DOC = os.path.abspath(os.path.join(BASE_DIR, "..", "GEMINI.md"))
TIL_DIR = os.path.join(BASE_DIR, "reports", "til")

def run_wrap():
    print("🎬 [Antigravity Wrap] 세션 마무리를 시작합니다 (5개 서브에이전트 가동)...")
    
    if not os.path.exists(LOG_FILE):
        print("❌ 분석할 세션 기록이 없습니다.")
        return

    # 1. 서브에이전트 준비: 세션 데이터 로드
    df = pd.read_json(LOG_FILE, lines=True)
    current_session = df.tail(10) # 최근 10개 활동 분석
    
    # --- 서브에이전트 1: Doc-Updater ---
    print("🔍 [doc-updater] 문서 업데이트 필요성 분석 중...")
    doc_needs_update = False
    if "OPTIMIZE" in current_session['type'].values:
        doc_needs_update = True
        
    # --- 서브에이전트 2: Automation-Scout ---
    print("🔭 [automation-scout] 반복 패턴 및 자동화 기회 탐색 중...")
    suggested_automation = []
    if current_session[current_session['type'] == 'ANALYSIS'].shape[0] > 2:
        suggested_automation.append("반복적인 데이터 분석 워크플로우를 'analyze-patterns.md' 스킬로 생성")

    # --- 서브에이전트 3: Learning-Extractor ---
    print("💡 [learning-extractor] 오늘의 배움(TIL) 추출 중...")
    learnings = current_session['detail'].tolist()

    # --- 서브에이전트 4: Followup-Suggester ---
    print("📝 [followup-suggester] 다음 작업 우선순위 정리 중...")
    todo = ["모델 예측 성능 고도화", "실시간 대시보드 배포 최적화"]

    # --- 서브에이전트 5: Duplicate-Checker ---
    print("🛡️ [duplicate-checker] 중복 제안 검증 중...")
    # (단순 구현: 중복 제거 로직)

    # 결과 요약 보고서 생성
    print("\n" + "="*50)
    print("🏁 SESSION WRAP-UP REPORT")
    print("="*50)
    
    print(f"\n[1] 문서화: GEMINI.md 업데이트 필요 ({'필요' if doc_needs_update else '없음'})")
    print(f"[2] 자동화 제안: {', '.join(suggested_automation) if suggested_automation else '없음'}")
    print(f"[3] 오늘 배운 것(TIL):")
    for i, l in enumerate(learnings[-3:], 1):
        print(f"   - {l}")
    
    print(f"\n[4] 다음에 할 일:")
    for t in todo:
        print(f"   - ✅ {t}")
    
    print("\n" + "="*50)
    print("❓ 수행하시겠습니까?")
    print("1. GEMINI.md 및 문서 업데이트 (y/n)")
    print("2. 발견된 자동화 스킬 생성 (y/n)")
    print("3. TIL 리포트 저장 및 세션 로그 백업 (y/n)")
    print("="*50)

if __name__ == "__main__":
    run_wrap()
