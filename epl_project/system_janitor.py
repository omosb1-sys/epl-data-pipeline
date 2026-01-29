import os
import time
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🧹 %(message)s')
logger = logging.getLogger("SystemJanitor")

class SystemJanitor:
    """
    프로젝트의 임시 파일, 로그, 오래된 리포트를 관리하는 시스템 환경미화원
    """
    def __init__(self, retention_days: int = 7):
        # epl_project 기준 경로 설정
        self.base_dir = Path(__file__).resolve().parent
        self.retention_seconds = retention_days * 24 * 60 * 60
        
        # 청소 대상 디렉토리 및 파일 패턴
        self.cleanup_targets = [
            # 1. 크몽 프로젝트 임시 가공 데이터 (Parquet)
            {"path": self.base_dir / "data" / "kmong_project" / "processed", "pattern": "*.parquet"},
            # 2. 분석 리포트 및 로그
            {"path": self.base_dir / "reports", "pattern": "*.*"},
            {"path": self.base_dir / "output", "pattern": "*.*"},
            # 3. 임시 요구사항 명세서 스냅샷 (만약 있다면)
            {"path": self.base_dir.parent, "pattern": "requirements_*.txt"}
        ]

    def clean_old_files(self):
        """설정된 유예 기간이 지난 파일 삭제"""
        print("\n" + " ✨ " * 10)
        logger.info(f"시스템 환경미화 가동 (유예 기간: {self.retention_seconds // 86400}일)")
        print(" ✨ " * 10 + "\n")

        now = time.time()
        deleted_count = 0
        freed_space = 0

        for target in self.cleanup_targets:
            target_path = target["path"]
            if not target_path.exists():
                continue

            for file_path in target_path.glob(target["pattern"]):
                # 파일 수정 시간 확인
                file_mtime = file_path.stat().st_mtime
                if (now - file_mtime) > self.retention_seconds:
                    try:
                        file_size = file_path.stat().st_size
                        file_path.unlink() # 파일 삭제
                        deleted_count += 1
                        freed_space += file_size
                        logger.info(f"임시 파일 삭제: {file_path.name} ({file_size / 1024:.1f} KB)")
                    except Exception as e:
                        logger.error(f"삭제 실패 ({file_path.name}): {e}")

        # 4. __pycache__ 정리
        self._clean_pycache()

        if deleted_count > 0:
            logger.info(f"청소 완료: 총 {deleted_count}개 파일 삭제 (약 {freed_space / (1024*1024):.2f} MB 확보)")
        else:
            logger.info("정리할 오래된 파일이 없습니다. 시스템이 쾌적합니다.")

    def _clean_pycache(self):
        """Python 컴파일 캐시 정리"""
        for pycache in self.base_dir.parent.glob("**/__pycache__"):
            try:
                import shutil
                shutil.rmtree(pycache)
                logger.info(f"파이썬 캐시 정리: {pycache.relative_to(self.base_dir.parent)}")
            except: pass

if __name__ == "__main__":
    # 7일 이상 된 파일 정리
    janitor = SystemJanitor(retention_days=7)
    janitor.clean_all = True # 추가 기능 확장용
    janitor.clean_old_files()
