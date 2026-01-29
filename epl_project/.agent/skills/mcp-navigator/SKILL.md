---
name: mcp-navigator
description: GitHub MCP 서버를 활용한 효율적인 태스크 관리 및 PAW 프로토콜 실행 지침
user-invocable: true
---

# 🐙 MCP Navigator: GitHub 전용 에이전트 워크플로우

본 지침은 `github` MCP 서버를 활용하여 `GEMINI.md`의 **Rule 40 (PAW Protocol)**을 물리적인 코드 관리 체계와 동기화하기 위한 가이드입니다.

## 🚀 1. PAW 기반 격리 작업 (Isolated Mission)
1.  **Branch Per Task**: 모든 신규 태스크(`TASK_BOARD.md`의 In Progress 상태)는 전용 브랜치를 생성하여 작업을 시작합니다.
2.  **Worktree Sync**: 로컬의 `git worktree` 격리 상태와 원격의 GitHub 브랜치 상태를 상시 동기화하여 다중 작업 시 충돌을 방지합니다.

## 🤝 2. 협업 및 검증 (Code Review Strategy)
1.  **Draft PR Flow**: 작업이 시작되면 즉시 Draft Pull Request를 생성하여 현재 진행 상황을 투명하게 공개합니다.
2.  **Automated Commenting**: 주요 마일스톤 달성 시 PR에 자동으로 진행 상황 요약 코멘트를 남겨 Rule 28.2(Artifact-Driven Validation)를 실현합니다.
3.  **Cross-Agent Review**: `Review Agent`가 PR 파일을 검토하고 승인(Approve)한 후에만 최종 병합(Merge)을 시도합니다.

## 🛡️ 3. 보안 가드레일 (Rule 39 준수)
1.  **Secret Scrubbing**: PR을 올리기 전, MCP를 통해 파일 내용을 전송할 때 하드코딩된 비밀 정보가 포함되지 않았는지 `security-guardian` 스킬로 최종 확인합니다.
2.  **Issue Tracking**: 모든 버그나 기술 부채는 GitHub Issue로 생성하여 장기적인 추적성을 확보합니다.

---
*Created by Antigravity (MCP & Engineering Orchestration Lab) - 2026.01.26*
