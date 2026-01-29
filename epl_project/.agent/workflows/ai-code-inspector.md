---
description: AI 코드 품질 검증 워크플로우
---

# AI Code Quality Inspector

## 목적
생성된 모든 Python 코드를 실행 전 AI가 자동으로 검증하여 품질 균일화

## 검증 단계

### 1. 정적 분석 (Static Analysis)
```bash
# Pylint로 코드 스타일 검사
pylint --rcfile=.pylintrc {생성된_파일}.py

# Mypy로 타입 체크
mypy --strict {생성된_파일}.py
```

### 2. 논리 검증 (Logic Verification)
- **변수 선언 전 사용 여부** 확인
- **함수 호출 시 인자 타입** 일치 여부
- **반환값 타입** 일치 여부

### 3. 성능 프로파일링 (Performance Check)
```python
# 예상 실행 시간 계산
import cProfile
cProfile.run('main()', sort='cumtime')
```

### 4. 보안 스캔 (Security Scan)
```bash
# Bandit로 보안 취약점 검사
bandit -r {생성된_파일}.py
```

## 자동화 트리거
- 모든 `write_to_file` 호출 후 자동 실행
- 검증 실패 시 사용자에게 경고 + 수정안 제시

## 예시 출력
```
✅ 정적 분석: 통과 (Pylint 9.2/10)
✅ 논리 검증: 통과
⚠️ 성능: Line 45에서 O(n²) 루프 감지 → 최적화 제안
✅ 보안: 취약점 없음
```
