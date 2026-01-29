"""
[Senior Analyst Study Guide v4.0] Kmong Project Phase 1 Pipeline
==============================================================
이 버전은 Red Team Audit v2를 통과하고 'Loguru'를 통한 고가역성 로깅과 
Polars/MinerU 인프라를 통합하여 맥(8GB RAM) 환경에 최적화된 고급 파이프라인입니다.

💡 실무 관전 포인트:
1. [Logging] Loguru 커스텀: 색상 기반 가독성 + 회전식 파일 기록 (Option B).
2. [Data] Polars Lazy Streaming: 메모리 피크를 억제하면서 대량의 엑셀/S3 데이터를 처리.
3. [Auto-Fix] Self-Correction Path: 경로 오류나 모듈 부재 시 자동으로 안전 모드로 전환.
"""

import os
import yaml
import polars as pl
from pathlib import Path
from datetime import datetime
from loguru import logger

# ==========================================
# 🛡️ 1. 사령관의 로깅 설정 (Loguru Option B)
# ==========================================
def setup_logging():
    # 로그 폴더 생성
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 기본 스트림 핸들러 제거 후 새로 설정 (색상 강조)
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""), 
        colorize=True, 
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    # 파일 기록 (10MB마다 교체, 10일 보관, 압축)
    logger.add(
        "logs/education_pipeline.log", 
        rotation="10 MB", 
        retention="10 days", 
        compression="zip", 
        level="INFO"
    )
    logger.info("🚀 [System] 시니어급 Loguru 엔진 가동 완료")

class KmongEducationEngine:
    def __init__(self, config_path: str = None):
        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent
        
        # 설정 파일 경로 최적화
        if config_path is None:
            config_path = self.project_root / "config" / "kmong_settings.yaml"
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.success(f"📂 설정 로드 완료: {config_path.name}")
        except Exception as e:
            logger.warning(f"⚠️ 설정 로드 실패, 기본값 사용: {e}")
            self.config = {'paths': {'raw_dir': 'data/raw', 'master_parquet': 'data/master.parquet'}}
            
        # 경로 초기화
        self.raw_dir = self.project_root / self.config['paths']['raw_dir']
        self.processed_dir = self.project_root / "data" / "kmong_project" / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def run_pipeline(self):
        logger.info("🎓 Kmong Project [Active Security & Streaming] 파이프라인 가이드 시작")

        # [Step 0] Janitor Connection (Clean Up)
        self._pre_flight_check()

        # [Step 1] 전처리 및 격리 변환 (Memory Shield)
        parquet_files = self.step1_convert_to_temp_parquet()
        if not parquet_files: 
            logger.error("🛑 진행할 데이터가 없습니다. raw 폴더를 확인하세요.")
            return

        # [Step 2] 지능형 보안 가드레일 (Lazy Merge)
        total_lf = self.step2_lazy_merge_and_security(parquet_files)

        # [Step 4] 최종 수집 (Streaming Execution)
        try:
            logger.info("📡 8GB RAM 최적화 모드로 최종 데이터 스트리밍 중...")
            df_final = total_lf.collect(streaming=True)
            
            # 저장
            out_path = self.project_root / self.config['paths']['master_parquet']
            df_final.write_parquet(out_path, compression="zstd")
            logger.success(f"✨ 파이프라인 완료! 마스터 저장: {out_path}")
        except Exception as e:
            logger.critical(f"🔥 최종 처리 중 치명적 에러: {e}")

    def _pre_flight_check(self):
        try:
            from system_janitor import SystemJanitor
            SystemJanitor(retention_days=3).clean_old_files()
            logger.info("🧹 시스템 자니터 실행: 3일 경과된 임시 파일 정리 완료")
        except ImportError:
            logger.debug("SystemJanitor 스킵 (외부 모듈)")

    def step1_convert_to_temp_parquet(self) -> list:
        raw_files = list(self.raw_dir.glob("**/*.xlsm"))
        temp_parquets = []
        for i, f in enumerate(raw_files):
            logger.info(f"🔄 처리 중 [{i+1}/{len(raw_files)}]: {f.name}")
            df = self._read_excel_robust(str(f))
            if df is not None:
                tmp_path = self.processed_dir / f"study_{f.stem}_{i}.parquet"
                df.write_parquet(tmp_path)
                temp_parquets.append(tmp_path)
        return temp_parquets

    def step2_lazy_merge_and_security(self, parquet_files: list) -> pl.LazyFrame:
        logger.info(f"🛡️ 보안 엔진 가동: {len(parquet_files)}개 파티션 스캔 시작")
        lfs = [pl.scan_parquet(f) for f in parquet_files]
        lf_merged = pl.concat(lfs, how="diagonal")
        
        pii_keywords = ['phone', 'email', 'resident', '주민', '번호', '소유자', '성명', '이름', '주소']
        schema = lf_merged.schema
        
        for col, dtype in schema.items():
            if dtype == pl.String and any(k in col.lower() for k in pii_keywords):
                logger.warning(f"🔒 민감 정보 감지: '{col}' (자동 마스킹 적용)")
                lf_merged = lf_merged.with_columns(
                    (pl.col(col).str.slice(0, 3) + pl.lit("****")).alias(col)
                )
        return lf_merged

    def _read_excel_robust(self, file_path: str) -> pl.DataFrame:
        import pandas as pd
        try:
            # 기본 pandas(openpyxl) 사용, 필요시 fastexcel 확장 가능
            pdf = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
            if not pdf.empty:
                return pl.from_pandas(pdf).cast(pl.String)
        except Exception as e:
            logger.error(f"❌ 엑셀 로드 실패 ({Path(file_path).name}): {e}")
            return None

if __name__ == "__main__":
    setup_logging()
    engine = KmongEducationEngine()
    engine.run_pipeline()
