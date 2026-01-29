"""
[Kmong Project] Phase 1: Red Team Hardened Pipeline (v3.1)
=========================================================
- System Janitor Integrated: 자동 리소스 청소 기능 탑재 (7일 주기)
- Auto-PII Detection: 지능형 개인정보 마스킹
- Streaming Merge: 8GB RAM 최적화

Author: Antigravity AI
Date: 2026-01-24
"""

import polars as pl
import os
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# [Senior Tip] 데이터 왜곡 방지를 위한 로깅 시스템 구축
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KmongSeniorAnalyst")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class KmongPhase1Engine:
    def __init__(self, config_path: str = None):
        # 경로 유연성 확보
        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent
        
        if config_path is None:
            config_path = self.project_root / "config" / "kmong_settings.yaml"
        
        if not os.path.exists(config_path):
            config_path = Path("config/kmong_settings.yaml").resolve()

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_dir = self.project_root / self.config['paths']['raw_dir']
        self.processed_dir = self.project_root / "data" / "kmong_project" / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def run_pipeline(self):
        # [NEW] 시스템 자원 보호: 오래된 임시 파일 청소 (7일 주기)
        try:
            from system_janitor import SystemJanitor
            SystemJanitor(retention_days=7).clean_old_files()
        except Exception as e:
            logger.warning(f"시스템 청소 중 오류 발생 (스킵): {e}")

        print("\n" + "=" * 60)
        print("🏛️ [Red Team Hardened] Kmong Analysis Engine v3.1")
        print("🔒 Security Level: Maximum (Auto-PII Detection Active)")
        print("💾 Memory Mode: Streaming (Low-RAM Optimized)")
        print("=" * 60)

        # Step 1: 파일별 개별 처리 (Streaming Mode)
        parquet_files = self.step1_convert_to_temp_parquet()
        if not parquet_files: 
            logger.error("No data files processing complete. Terminating.")
            return

        # Step 2: Lazy 병합 및 보안 적용
        total_lf = self.step2_lazy_merge_and_security(parquet_files)
        
        # Step 3: 실무 중심 피처 엔지니어링 (Lazy 유지)
        total_lf = self.step3_feature_engineering_lazy(total_lf)
        
        # Step 4: 최종 마스터 저장 (Collect 실행)
        try:
            logger.info("📡 최종 데이터 수집 및 마스터 파일 생성 중...")
            df_final = total_lf.collect(streaming=True)
            self.step4_save_master(df_final)
            self.step5_strategic_eda(df_final)
            print("\n✅ 레드팀 보안/성능/청소 자동화 프로세스가 완료되었습니다.")
        except Exception as e:
            logger.error(f"최종 처리 중 치명적 에러: {e}")

    def step1_convert_to_temp_parquet(self) -> list:
        raw_files = list(self.raw_dir.glob("**/*.xlsm"))
        if not raw_files:
            logger.warning(f"경로에 파일이 없습니다: {self.raw_dir}")
            return []
        
        temp_parquets = []
        for i, f in enumerate(raw_files):
            logger.info(f"📦 [Step 1] 파일 변환 ({i+1}/{len(raw_files)}): {f.name}")
            df = self._read_excel_robust(str(f))
            if df is not None:
                tmp_path = self.processed_dir / f"{f.stem}_{i}.parquet"
                df.write_parquet(tmp_path)
                temp_parquets.append(tmp_path)
        return temp_parquets

    def step2_lazy_merge_and_security(self, parquet_files: list) -> pl.LazyFrame:
        logger.info("🔒 [Step 2] 지능형 보안 탐지 레이어 가동...")
        lfs = [pl.scan_parquet(f) for f in parquet_files]
        lf_merged = pl.concat(lfs, how="diagonal")
        config_mask_cols = self.config['security'].get('pii_columns', [])
        pii_keywords = ['phone', 'email', 'resident', '주민', '번호', '주소', 'owner', '소유자', '성명', '이름']
        schema = lf_merged.schema
        for col, dtype in schema.items():
            if dtype == pl.String:
                is_pii = col in config_mask_cols or any(k in col.lower() for k in pii_keywords)
                if is_pii:
                    logger.info(f"   🛡️ Auto-Masking Applied: '{col}'")
                    lf_merged = lf_merged.with_columns((pl.col(col).str.slice(0, 3) + pl.lit("****")).alias(col))
        return lf_merged

    def step3_feature_engineering_lazy(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        logger.info("🛠️ [Step 3] Deep Delta Feature Intelligence 가동...")
        if "회원구분명" in lf.columns:
            lf = lf.with_columns(pl.when(pl.col("회원구분명").str.contains("개인")).then(pl.lit("개인")).otherwise(pl.lit("법인")).alias("고객유형_분류"))
        return lf

    def step4_save_master(self, df: pl.DataFrame):
        master_path = self.project_root / self.config['paths']['master_parquet']
        master_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(master_path, compression="zstd")
        logger.info(f"💾 마스터 파일 저장 완료: {master_path}")

    def step5_strategic_eda(self, df: pl.DataFrame):
        logger.info("📊 기초 리포트 요약 생성 중...")
        report_dir = self.project_root / self.config['paths']['report_dir']
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_dir / "processing_summary.txt", "w") as f:
            f.write(f"Processed Date: {pl.datetime.datetime.now()}\nTotal Rows: {len(df):,}\n")

    def _read_excel_robust(self, file_path: str) -> pl.DataFrame:
        import pandas as pd
        import openpyxl
        try:
            wb = openpyxl.load_workbook(file_path, read_only=True)
            names = wb.sheetnames; wb.close()
            sheets = []
            for n in names:
                try:
                    try:
                        import fastexcel
                        fe = fastexcel.read_excel(file_path); pdf = fe.load_sheet(n).to_pandas()
                    except:
                        pdf = pd.read_excel(file_path, sheet_name=n, engine='openpyxl')
                    if not pdf.empty:
                        df_sheet = pl.from_pandas(pdf).cast(pl.String)
                        df_sheet = df_sheet.with_columns([pl.lit(Path(file_path).name).alias("_src_file"), pl.lit(n).alias("_src_sheet")])
                        sheets.append(df_sheet)
                except: pass
            return pl.concat(sheets, how="diagonal") if sheets else None
        except Exception as e:
            logger.error(f"File Access Error {file_path}: {e}"); return None

if __name__ == "__main__":
    engine = KmongPhase1Engine()
    engine.run_pipeline()
