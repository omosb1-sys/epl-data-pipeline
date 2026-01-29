"""
Mac System Care Agent (Lightweight MCP Prototype) - Version 1.1
사용자님의 Intel Mac (8GB RAM) 환경을 쾌적하게 유지하기 위한 캐시 및 리소스 정리 도구.
(권한 오류 방지를 위해 에러 핸들링 보강)
"""

import os
import shutil
import subprocess
from pathlib import Path

class MacCareAgent:
    def __init__(self):
        self.home = Path.home()
        self.cache_paths = [
            self.home / "Library/Caches",
            self.home / "Library/Logs",
            Path("/Library/Caches"),
        ]

    def check_resource_status(self):
        """현재 맥의 RAM 및 디스크 상태 확인"""
        print("📊 [System Monitor] 리소스 상태 점검 중...")
        try:
            # PhysMem만 추출하여 간결하게 출력
            ram_info = subprocess.check_output("top -l 1 -s 0 -n 0 | grep PhysMem", shell=True).decode().strip()
            print(f"🧠 RAM 상태: {ram_info}")
        except Exception as e:
            print(f"🧠 RAM 상태 확인 실패: {e}")

    def analyze_cache_usage(self):
        """캐시 폴더 용량 분석 (권한 에러 무시)"""
        print("🔍 [Cache Analysis] 주요 캐시 경로 분석 중...")
        report = []
        for path in self.cache_paths:
            if path.exists():
                try:
                    # stderr=subprocess.DEVNULL로 권한 오류 메시지 차단
                    size = subprocess.check_output(["du", "-sh", str(path)], stderr=subprocess.DEVNULL).decode().split()[0]
                    report.append(f"- {path}: {size}")
                except Exception:
                    report.append(f"- {path}: Access Restricted (Partially analyzed)")
        return report

    def suggest_cleanup(self):
        """안전하게 삭제 가능한 항목 제안"""
        print("🧹 [Maintenance] 안전 청소 가이드...")
        
        # 1. 개발 관련 캐시
        npm_cache = self.home / ".npm/_cacache"
        if npm_cache.exists():
            print(f"✨ [.npm] 캐시가 용량을 차지하고 있습니다.")
            
        # 2. 브라우저 캐시 안내
        print("✨ [Safari/Chrome] 브라우저 설정에서 '캐시 비우기'를 실행하면 램 확보에 큰 도움이 됩니다.")

    def purge_inactive_memory(self):
        """비활성 메모리 강제 정리 (Intel Mac 특화)"""
        print("🚀 [RAM Optimizer] 비활성 메모리 정리 시도 (Purge)...")
        try:
            # purge는 macOS 내장 명령어로 비활성 메모리를 해제합니다.
            result = subprocess.run(["purge"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ 비활성 메모리가 성공적으로 반환되었습니다.")
            else:
                print("⚠️  메모리 정리 실패: 시스템 권한(Sudo)이 필요할 수 있습니다.")
        except FileNotFoundError:
            print("⚠️  'purge' 명령어를 찾을 수 없습니다. (CommandLine Tools 필요)")

if __name__ == "__main__":
    care = MacCareAgent()
    care.check_resource_status()
    usage = care.analyze_cache_usage()
    for line in usage:
        print(line)
    care.suggest_cleanup()
    care.purge_inactive_memory()
