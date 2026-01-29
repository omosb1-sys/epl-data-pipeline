
"""
[Antigravity Zero-Static Credential Scanner]
Inspired by Amazon IAM Best Practices
============================================
이 도구는 Antigravity Rule 39를 준수하여, 프로젝트 내에 하드코딩된
AWS Access Key, Secret Key 등의 민감한 정적 자격 증명이 있는지 스캔합니다.
"""

import re
import os
from pathlib import Path
from loguru import logger

# [Rule 39.4] 탐지 대상 정규표현식 패턴 (AWS 자격 증명 표준 패턴)
PATTERNS = {
    "AWS_ACCESS_KEY_ID": re.compile(r"([^A-Z0-9]|^)(AKIA[0-9A-Z]{16})([^A-Z0-9]|$)"),
    "AWS_SECRET_ACCESS_KEY": re.compile(r"([^A-Za-z0-9/+=]|^)([A-Za-z0-9/+=]{40})([^A-Za-z0-9/+=]|$)"),
}

def scan_file(file_path: Path):
    """파일 내에서 정적 키 노출 여부를 검사합니다."""
    findings = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, 1):
                for key_type, pattern in PATTERNS.items():
                    if pattern.search(line):
                        findings.append((line_no, key_type))
    except Exception as e:
        logger.error(f"Failed to scan {file_path}: {e}")
    return findings

def run_security_audit(project_root: str):
    root = Path(project_root)
    logger.info(f"🛡️ [Security Audit] '{root.name}' 정적 자격 증명 스캔 시작...")
    
    total_findings = 0
    # 스캔 대상 확장자 제한 (소스코드 및 설정 파일)
    target_extensions = {".py", ".env", ".yaml", ".yml", ".json", ".sh"}
    
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in target_extensions:
            # .agent 폴더 등 내부 폴더 제외
            if ".agent" in path.parts or ".venv" in path.parts:
                continue
                
            findings = scan_file(path)
            if findings:
                for line_no, key_type in findings:
                    logger.critical(f"🚨 [LEAK DETECTED] {path.relative_to(root)}:{line_no} - {key_type} 노출!")
                    total_findings += 1
                    
    if total_findings == 0:
        logger.success("✨ [Security Audit] 정적 자격 증명 노출이 발견되지 않았습니다. (Zero-Static Credential 준수)")
    else:
        logger.warning(f"📉 [Security Audit] 총 {total_findings}건의 보안 위반 항목이 발견되었습니다. 즉시 조치(Rule 39.2)가 필요합니다.")

if __name__ == "__main__":
    # 프로젝트 루트(epl_project) 폴더를 대상으로 스캔 시뮬레이션
    project_path = Path(__file__).parent
    run_security_audit(str(project_path))
