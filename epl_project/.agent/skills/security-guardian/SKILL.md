---
name: security-guardian
description: AI 사용 시 데이터 유출 방지 및 보안 가드레일을 강제하는 보안 관리 스킬 (Rule 15, 39 통합)
user-invocable: true
---

# 🛡️ Security Guardian: 엔터프라이즈급 보안 및 개인정보 보호 프로토콜

이 스킬은 "AI가 정말 안전한가?"라는 사용자의 우려에 답하고, `GEMINI.md`의 **Rule 39 (Zero-Static Credentials)**를 실무적으로 강제하기 위한 안티그래비티의 핵심 보안 엔진입니다.

## 🔐 1. 데이터 비식별화 및 익명화 (Local De-identification)
1.  **PII Scrubbing**: 실제 데이터를 다루기 전, 이름, 전화번호, 이메일, 상세 주소 등 개인 식별 정보(PII)를 로컬에서 반드시 마스킹하거나 삭제합니다. (kmong 프로젝트 데이터 가이드 준수)
2.  **Dummy Data First**: 로직 검증이나 코딩 질문 시에는 실제 데이터가 아닌, 구조만 동일한 **가상 데이터(Dummy Data)**를 생성하여 AI 컨텍스트에 주입합니다.
3.  **On-device Metadata Extraction**: 뉴스나 리뷰 등 비정형 데이터 분석 시, 로컬 프로세스에서 핵심 메타데이터만 추출하고 민감한 원문 전체 전송은 지양합니다.

## 🔑 2. Zero-Static Credentials (Rule 39 강제)
1.  **No Hardcoded Secrets**: AWS Access Key, GCP Key 등을 코드나 `.env`, `.yaml`에 직접 노출하는 행위를 엄격히 금지합니다.
2.  **Identity-Based Security**: 장기 키 대신 IAM Role, OIDC 또는 임시 자격 증명(STS) 사용을 유도합니다.
3.  **Credential Scanning**: 모든 작업 전 `security_credential_scanner.py`를 가동하여 하드코딩된 비밀 키가 시스템에 유입되는 것을 선제적으로 차단합니다.

## ⚙️ 3. 안티그래비티 보안 가드레일 (Rule 15 연계)
1.  **Zero-Trust Permission**: 파일 수정이나 터미널 명령 실행 전, 반드시 **'작업 영향 보고(Impact Statement)'**를 생성하고 사용자의 승인을 받습니다.
2.  **Telemetry Off**: 데이터가 모델 학습에 사용되지 않도록 설정 상태를 상시 감시합니다.
3.  **Log Pruning**: 보안 민감 로그는 작업 완료 후 Rule 15.4에 따라 즉시 삭제(Pruning)하여 잔상을 남기지 않습니다.

---
*Updated by Antigravity (Enterprise Security & Privacy Refinement) - 2026.01.26*
