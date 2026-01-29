
"""
[Antigravity SRE Health Monitor v1.0]
Inspired by Swizec Teller's 'Operational Excellence'
===================================================
이 가이드는 Antigravity Rule 37(SRE Philosophy)을 실무적으로 구현하여,
시스템이 사용자에게 보고하기 전 스스로의 건강 상태를 진단하고 복구하는 로직을 시뮬레이션합니다.
"""

import os
import psutil
import shutil
from pathlib import Path
from loguru import logger
import time

class AntigravitySRE:
    def __init__(self, project_root: str):
        self.root = Path(project_root)
        self.log_file = self.root / "logs" / "education_pipeline.log"
        
    def check_disk_space(self, threshold_gb: float = 1.0):
        """[Rule 37.1] Disk Observability"""
        total, used, free = shutil.disk_usage(self.root)
        free_gb = free / (2**30)
        
        if free_gb < threshold_gb:
            logger.warning(f"🚨 [SRE Alert] 부족한 디스크 공간: {free_gb:.2f}GB (임계값: {threshold_gb}GB)")
            return False
        logger.info(f"✅ [SRE Health] 디스크 공간 양호: {free_gb:.2f}GB 여유")
        return True

    def check_log_health(self):
        """[Rule 37.2] Proactive Log Analysis"""
        if not self.log_file.exists():
            logger.error("🛑 [SRE Alert] 로그 파일이 존재하지 않습니다. 파이프라인 중단 가능성.")
            return False
        
        # 최근 10줄에서 ERROR 키워드 검색
        with open(self.log_file, "r") as f:
            lines = f.readlines()[-10:]
            errors = [l for l in lines if "ERROR" in l or "CRITICAL" in l]
            
        if errors:
            logger.warning(f"⚠️ [SRE Performance] 최근 로그에서 {len(errors)}건의 에러 감지. 자가 회복 확인 필요.")
            return False
        logger.info("✅ [SRE Health] 최근 로그 깨끗함.")
        return True

    def run_health_check(self):
        logger.info("🚀 [Antigravity SRE] 시스템 통합 상태 진단 시작...")
        results = {
            "disk": self.check_disk_space(),
            "logs": self.check_log_health()
        }
        
        if all(results.values()):
            logger.success("✨ [SRE Status] 모든 시스템 정상 운영 중 (Operational Excellence)")
        else:
            logger.warning("📉 [SRE Status] 일부 시스템 주의 필요 - Healing Agent 준비")

if __name__ == "__main__":
    # 프로젝트 루트 설정 (현재 폴더 기준)
    project_path = Path(__file__).parent
    sre = AntigravitySRE(project_path)
    sre.run_health_check()
