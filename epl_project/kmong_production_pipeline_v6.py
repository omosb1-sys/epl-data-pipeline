"""
[Kmong Project] Production Pipeline v6.0 (Quantum Leap - ML Readiness)
=====================================================================
- Rule 8.1: Advanced Feature Discretization & Log Transformation
- Rule 17: Multi-stage Quality Scaffolding (DQS)
- Rule 33: Russ Cox Protocol (High-Precision Scaling)
- Rule 38: Types as Proofs (Analytical Integrity)
- ML/Statistics: Outlier Management (IQR), Skewness Correction, Robust Scaling

Author: Antigravity AI (Senior Analyst)
Date: 2026-01-26
"""

import polars as pl
import os
import yaml
import gc
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from loguru import logger

def setup_logging():
    """고해상도 로깅 설정"""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""), 
        colorize=True, 
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    logger.add(
        "logs/kmong_production_v6.log", 
        rotation="10 MB", 
        retention="10 days", 
        compression="zip", 
        level="INFO"
    )

class KmongQuantumPipeline:
    """고급 통계 지능이 탑재된 6세대 데이터 자산화 파이프라인"""
    
    def __init__(self, config_path: str = "config/kmong_settings_v6.yaml"):
        self.project_root = Path(__file__).resolve().parent
        self.config_path = self.project_root / config_path
        self.load_config()
        self.start_time = datetime.now()
        self.metadata = {"pipeline_version": "6.0", "stats": {}}
        
    def load_config(self) -> None:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.success(f"✅ V6 Analytics Config Loaded: {self.config_path.name}")
        except Exception as e:
            logger.error(f"⚠️ Config error: {e}. Falling back to default.")
            self.config = {'paths': {'raw_dir': 'data/kmong_project/raw', 'processed_dir': 'data/kmong_project/processed_v6'}}

    def run(self) -> None:
        logger.info(f"🏗️ [Quantum Leap] Starting Advanced ML-Ready Pipeline")
        try:
            # 1. 수집 및 기초 가공 (Streaming)
            temp_parquets = self.ingest_and_clean()
            if not temp_parquets: return

            # 2. 보안 레이어 (Rule 39)
            lf = self.apply_security(temp_parquets)

            # 3. 고급 통계 분석 레이어 (Statistical Intelligence)
            lf, stats_report = self.apply_statistical_intelligence(lf)
            self.metadata["stats"] = stats_report

            # 4. ML 특징 공학 레이어 (Feature Engineering)
            lf = self.apply_ml_features(lf)

            # 5. 최종 데이터 자산화 (Assetization)
            self.finalize_asset(lf)

            # 6. 품질 품질 게이트 (Data Quality Score)
            self._execute_quality_gate()

            logger.success(f"🏆 Quantum Leap complete. Analytics metadata saved.")
        except Exception as e:
            logger.critical(f"🔥 Pipeline crash: {e}")
            raise

    def ingest_and_clean(self) -> List[Path]:
        raw_dir = self.project_root / self.config['paths']['raw_dir']
        processed_dir = self.project_root / self.config['paths']['processed_dir']
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        raw_files = list(raw_dir.glob("**/*.xlsm")) + list(raw_dir.glob("**/*.xlsx"))
        temp_paths = []
        
        for i, f in enumerate(raw_files):
            logger.info(f"📦 [Ingest] {f.name}")
            # [Senior Tip] 데이터 읽기 전 메모리 플러싱
            gc.collect()
            
            try:
                import fastexcel
                excel = fastexcel.read_excel(f)
                sheets = [pl.from_pandas(excel.load_sheet(sn).to_pandas()).cast(pl.String) for sn in excel.sheet_names]
                df = pl.concat(sheets, how="diagonal")
                
                # 기초 스키마 강제 (문자열 정규화)
                df = df.with_columns(pl.all().str.strip_chars())
                
                out_path = processed_dir / f"v6_tmp_{f.stem}_{i}.parquet"
                df.write_parquet(out_path)
                temp_paths.append(out_path)
            except Exception as e:
                logger.error(f"❌ Ingest fail: {f.name} -> {e}")
                
        return temp_paths

    def apply_security(self, parquet_files: List[Path]) -> pl.LazyFrame:
        lf = pl.concat([pl.scan_parquet(f) for f in parquet_files], how="diagonal")
        pii_cols = self.config['security'].get('pii_columns', [])
        
        # Rule 39: 익명화 및 추적 식별자 주입
        for col in lf.columns:
            if any(k in col.lower() for k in pii_cols) or any(k in col.lower() for k in ['성명', '주소', '전화']):
                lf = lf.with_columns((pl.col(col).str.slice(0, 2) + pl.lit("***")).alias(col))
        return lf

    def apply_statistical_intelligence(self, lf: pl.LazyFrame) -> Tuple[pl.LazyFrame, Dict]:
        """[고급 통계] Outlier 탐지 및 Skewness 분석"""
        logger.info("🧪 [Intelligence] Statistical profiling starting...")
        
        # 분석 대상 수치형 컬럼 추출
        target_cols = self.config['analysis'].get('ml_ready_columns', [])
        stats_report = {}

        for col in target_cols:
            if col in lf.columns:
                # 1. 데이터 클렌징 (숫자화) 및 결측치 처리 (Mean 전략 적용)
                lf = lf.with_columns(
                    pl.col(col).str.replace_all("[^0-9.]", "").cast(pl.Float64, strict=False).fill_null(strategy="mean")
                )
                
                # 2. 통계 지표 계산 (Lazy 상태에서는 직접 계산이 어려우므로 샘플 추출 후 계산)
                # [Senior Logic] 대규모 데이터의 경우 샘플링 기반 통계 추정
                sample_df = lf.select(col).limit(10000).collect()
                series = sample_df[col]
                
                # 데이터가 비어있거나 모두 Null인 경우 스킵
                if series.null_count() == len(series) or len(series) == 0:
                    logger.warning(f"  ⚠️ Skipping stats for '{col}': No valid data.")
                    continue

                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                
                # NoneType 방어
                if q1 is None or q3 is None:
                    logger.warning(f"  ⚠️ Quantile calculation returned None for '{col}'.")
                    continue

                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Skewness (왜곡도)
                skew = series.skew()
                if skew is None: skew = 0.0
                
                stats_report[col] = {
                    "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
                    "lower_bound": float(lower_bound), "upper_bound": float(upper_bound),
                    "skewness": float(skew)
                }
                
                # 3. 이상치 플래그 주입 (ML 모델에게 이상치 존재 여부를 알림)
                lf = lf.with_columns(
                    pl.when((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound))
                    .then(pl.lit(1)).otherwise(pl.lit(0)).alias(f"{col}_is_outlier")
                )

        return lf, stats_report

    def apply_ml_features(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """[ML 정확도 향상] 로그 변환 및 Robust Scaling 아키텍처"""
        logger.info("🏗️ [Features] ML-Ready transformation (Log & Robust) path...")
        
        target_cols = self.config['analysis'].get('ml_ready_columns', [])
        skew_threshold = self.config['analysis'].get('skewness_threshold', 0.75)

        for col in target_cols:
            if col in self.metadata["stats"]:
                col_stats = self.metadata["stats"][col]
                
                # 1. 자동 로그 변환 (Skewness 극복)
                if abs(col_stats["skewness"]) > skew_threshold:
                    logger.info(f"   📈 Log-Transform applied to '{col}' (Skew: {col_stats['skewness']:.2f})")
                    lf = lf.with_columns(
                        (pl.col(col) + 1).log().alias(f"{col}_log")
                    )
                
                # 2. Robust Scaling (IQR 기반)
                # 공식: (x - Q2) / (Q3 - Q1)
                median_val = col_stats["q1"] + (col_stats["iqr"] / 2) # 근사값
                if col_stats["iqr"] > 0:
                    lf = lf.with_columns(
                        ((pl.col(col) - median_val) / col_stats["iqr"]).alias(f"{col}_robust_scaled")
                    )
        
        # 3. 시간 피처 (Seasonality)
        if "등록일자" in lf.columns:
            lf = lf.with_columns(
                pl.col("등록일자").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
            ).with_columns([
                pl.col("등록일자").dt.month().alias("reg_month"),
                pl.col("등록일자").dt.weekday().alias("reg_dow")
            ])

        return lf

    def finalize_asset(self, lf: pl.LazyFrame) -> None:
        """최종 데이터 및 메타데이터 저장"""
        master_path = self.project_root / self.config['paths']['master_parquet']
        metadata_path = self.project_root / self.config['paths']['metadata_json']
        
        logger.info(f"📡 [Assetization] Writing master file: {master_path.name}")
        df = lf.collect(streaming=True)
        df.write_parquet(master_path, compression="zstd")
        
        with open(metadata_path, "w", encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=4, ensure_ascii=False)

    def _execute_quality_gate(self) -> None:
        """[Rule 17] Data Quality Score (DQS) 평가"""
        master_path = self.project_root / self.config['paths']['master_parquet']
        df = pl.read_parquet(master_path, n_rows=1000)
        
        # 품질 점수 계산 (가중치 기반)
        scores = {
            "Completeness": (1 - df.null_count().sum().to_numpy().sum() / (df.width * df.height)) * 100,
            "ML_Ready": (len([c for c in df.columns if "log" in c or "scaled" in c]) / len(self.config['analysis']['ml_ready_columns'])) * 100,
            "Security": 100 if "차대번호" in df.columns and "***" in str(df["차대번호"][0]) else 0
        }
        
        avg_score = sum(scores.values()) / len(scores)
        self.metadata["quality_score"] = avg_score
        
        logger.info(f"📊 [DQS Report] Total Quality Score: {avg_score:.1f}/100")
        for k, v in scores.items():
            logger.info(f"   - {k}: {v:.1f}")

if __name__ == "__main__":
    setup_logging()
    pipeline = KmongQuantumPipeline()
    pipeline.run()
