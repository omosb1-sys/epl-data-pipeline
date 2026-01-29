# 🔒 크몽 프로젝트 보안 체크리스트

**프로젝트**: 차량 등록 데이터 분석 (240만 행)  
**의뢰인**: 상무님 보고용  
**보안 등급**: ⚠️ 높음 (공공데이터이나 비식별 처리 필요)

---

## ✅ 작업 전 필수 체크

### 1. 데이터 수령 시
- [ ] 샘플 데이터 2개 확인 (2024.01, 2025.12)
- [ ] 개인정보 포함 여부 확인
- [ ] 비식별 처리 상태 확인
- [ ] 파일 암호화 여부 확인

### 2. 작업 환경 설정
- [x] `.gitignore` 설정 완료
- [ ] 크몽 전용 폴더 생성 (`data/kmong_project/`)
- [ ] Git 커밋 전 민감 데이터 제외 확인
- [ ] 환경 변수 분리 (`.env` 사용)

### 3. 코드 작성 시
- [ ] 하드코딩된 경로 금지 (상대 경로 사용)
- [ ] 샘플 데이터로 테스트 후 실제 데이터 적용
- [ ] 로그 파일에 민감 정보 출력 금지
- [ ] 에러 메시지에 데이터 노출 방지

### 4. 결과물 전달 시
- [ ] 원본 데이터 제거 (Parquet/Excel만 전달)
- [ ] 코드에서 경로 정보 제거
- [ ] PPT에 민감 정보 미포함 확인
- [ ] 최종 파일 압축 및 암호화

---

## 🚨 절대 금지 사항

### ❌ GitHub에 업로드 금지
```
data/kmong_project/
*.xlsm
*.xlsx
*.csv
*.parquet
```

### ❌ AI에게 직접 전달 금지
- 실제 고객 데이터 복붙
- 개인정보가 포함된 샘플
- 회사명/프로젝트명이 노출된 코드

### ❌ 외부 공유 금지
- 크몽 채팅 외 다른 채널로 데이터 전송
- 개인 클라우드(Dropbox, Google Drive)에 업로드
- SNS/블로그에 프로젝트 언급

---

## 🛡️ 안전한 작업 방법

### 1. 샘플 데이터 생성
```python
# 실제 데이터 대신 구조만 같은 더미 데이터 생성
import polars as pl

# 실제 데이터 로드
df_real = pl.read_excel("data/kmong_project/raw/sample.xlsx")

# 더미 데이터 생성 (구조만 복사)
df_dummy = df_real.head(5).with_columns([
    pl.lit("차량A").alias("차량명"),
    pl.lit("서울").alias("지역"),
    pl.lit("개인").alias("회원구분명")
])

# AI에게 질문할 때는 이 더미 데이터 사용
df_dummy.write_csv("data/sample_structure.csv")
```

### 2. 환경 변수 사용
```python
# .env 파일
KMONG_DATA_PATH=/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/data/kmong_project

# Python 코드
import os
from dotenv import load_dotenv

load_dotenv()
data_path = os.getenv("KMONG_DATA_PATH")
```

### 3. Git 커밋 전 확인
```bash
# 커밋 전 민감 데이터 체크
git status
git diff

# 실수로 추가된 파일 제거
git rm --cached data/kmong_project/*.xlsx
```

---

## 📋 작업 완료 후 체크

### 1. 로컬 정리
- [ ] 원본 데이터 백업 후 삭제
- [ ] 임시 파일 삭제 (`*.tmp`, `*.cache`)
- [ ] 로그 파일 정리

### 2. 전달 파일 검증
- [ ] 파일명에 민감 정보 없음
- [ ] 메타데이터 제거 (Excel 속성 정보)
- [ ] 압축 파일 암호 설정

### 3. 사후 관리
- [ ] 프로젝트 종료 후 30일 내 데이터 완전 삭제
- [ ] 크몽 채팅 기록 보관 (분쟁 대비)
- [ ] 포트폴리오 작성 시 데이터 익명화

---

## 🔐 Antigravity AI 보안 설정

### 현재 상태
- [x] `.gitignore` 설정 완료
- [ ] Telemetry 비활성화 (AI 학습 방지)
- [ ] Secure Mode 활성화
- [ ] Terminal Policy: Request Review

### 설정 방법
1. **File > Preferences > Antigravity Settings**
2. **Account 탭**
   - [ ] "Enable Telemetry" 체크 해제
3. **Security 탭**
   - [ ] "Secure Mode" 활성화
   - [ ] "Terminal Policy" → "Request Review"

---

## 📞 문제 발생 시

### 데이터 유출 의심 시
1. 즉시 작업 중단
2. 의뢰인에게 상황 보고
3. Git 히스토리 확인 및 민감 커밋 제거
4. 비밀번호/API 키 즉시 변경

### 의뢰인 문의 시
- "데이터는 로컬 PC에서만 처리됩니다"
- "작업 완료 후 30일 내 완전 삭제합니다"
- "제3자와 공유하지 않습니다"

---

**Last Updated**: 2026-01-22  
**Project**: 크몽 차량 등록 데이터 분석  
**Security Level**: HIGH
