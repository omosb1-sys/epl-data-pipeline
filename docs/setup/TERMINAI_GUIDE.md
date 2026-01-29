# 🤖 TerminaI Project Management Guide

이 파일은 **TerminaI**(로컬 시스템 오퍼레이터)가 이 프로젝트를 관리할 때 참조하는 지침서입니다.

## 1. 프로젝트 개요
- **목표**: EPL 데이터 수집 및 AI 기반 경기 예측 분석
- **핵심 파일**: `epl_project/collect_data.py`, `epl_project/app.py`

## 2. TerminaI 수행 임무 (Instructions)

### ① 자가 치유 (Self-Healing)
- 만약 `epl_project/internal/last_error.json` 파일이 생성되어 있다면, 즉시 내용을 읽고 에러 원인을 파악하세요.
- 파악된 원인에 따라 `collect_data.py`를 수정하거나 필요한 라이브러리를 설치한 후 다시 실행(`python epl_project/collect_data.py`)을 시도하세요.

### ② 인프라 제어 (Infrastructure as Language)
- 사용자가 "포트를 바꿔줘"라고 하면 `🚀_EPL_무료버전_실행하기.command` 파일 내의 `PORT` 변수를 수정하거나, 실행 시 `export EPL_PORT=8504`를 수행하도록 명령어를 조합하세요.

### ③ 데이터 무결성 검사 (AI-QA)
- `epl_project/internal/golden_set.json`의 고난도 케이스들을 기반으로, 현재 예측 모델이 논리적인 변수(체력, 로테이션, 더비 경기)를 제대로 반영하고 있는지 테스트 코드를 작성하여 검증하세요.

## 3. 명령 예시
- "지금 에러 난 거 수정하고 데이터 다시 받아와"
- "서버 포트 8888로 변경해서 다시 띄워"
- "골든 셋 테스트 결과 보고해줘"### ④ 세션 마무리 (/wrap)
- 사용자가 "세션 마무리해줘" 또는 "/wrap"이라고 말하면, `python epl_project/agent_wrap.py`를 실행하세요.
- 실행 결과로 나온 리포트를 사용자에게 요약해서 브리핑하고, 제안된 작업들(문서 업데이트, 스킬 생성 등)을 수행할지 물어보세요.
- 사용자가 승인하면 해당 작업을 저(Antigravity)와 협력하여 즉시 실행하세요.
