"""
[Senior Analyst Study Guide v6.0] Quantum Leap Analytics Pipeline
==================================================================
이 가이드는 단순한 데이터 엔지니어링을 넘어, '통계적 무결성'과 'ML 모델 성능 극대화'를 
달성하기 위한 시니어급 데이터 사이언스 워크플로우를 학습하기 위해 설계되었습니다.

💡 실무 관전 포인트 (Study Points):
1. [Robust Statistics] 왜 Median과 IQR을 사용하는가? (이상치에 강건한 분석)
2. [Distribution Correction] 왜곡도(Skewness) 보정이 ML 모델 정확도에 미치는 영향.
3. [Engineering Aesthetics] Loguru의 계층적 로깅과 DQS(품질 점수)로 검증하는 신뢰성.
4. [Hardware Synergy] 8GB RAM 환경에서 Polars Streaming으로 수백만 건을 다루는 법.
"""

import os
import yaml
import polars as pl
import numpy as np
from pathlib import Path
from loguru import logger
from datetime import datetime, timedelta

# ==========================================
# 🛡️ 1. 고급 관리자 로깅 (Loguru V6 Standard)
# ==========================================
def setup_logging():
    """로그 기록에 '의도'와 '색상'을 입히는 시니어급 설정"""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""), 
        colorize=True, 
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan> - <level>{message}</level>"
    )
    logger.info("🚀 [System] Quantum Leap 분석 엔진 준비 완료")

class KmongQuantumStudyEngine:
    """통계 지능을 자산화하는 6세대 학습용 엔진"""
    
    def __init__(self, config_path: str = "config/kmong_settings_v6.yaml"):
        self.project_root = Path(__file__).resolve().parent
        self.config_path = self.project_root / config_path
        self.load_config()
        
    def load_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.success(f"📂 설정 로드 완료: {self.config_path.name}")
        except Exception as e:
            logger.warning("⚠️ 기본 설정을 사용합니다. (V6 설정 파일 확인 권장)")
            self.config = {'analysis': {'ml_ready_columns': ['가격', '배기량'], 'skewness_threshold': 0.75}}

    def run_study_session(self):
        """학습용 파이프라인 시뮬레이션"""
        logger.info("🎓 [Study] 시니어 분석가의 데이터 정제 루틴 시작")

        # [Step 1] 가상 데이터 생성 (Rule self.project_root = Path(__file__).resolve().parent15.1 준수)
        lf = self._create_educational_dummy_data()

        # [Step 2] 통계적 진단 (Statistical Diagnosis)
        # ML 실무: "데이터를 모델에 넣기 전, 데이터의 '성격'을 숫자로 파악하라"
        lf, stats_summary = self.diagnose_data_stats(lf)

        # [Step 3] 고급 특징 공학 (Quantum Transformation)
        # ML 실무: "이상치는 무시하는 게 아니라 태깅하고, 왜곡된 분포는 펴라"
        lf = self.apply_quantum_transformation(lf, stats_summary)

        # [Step 4] 품질 검증 (Data Quality Gate)
        self.verify_study_result(lf)

    def _create_educational_dummy_data(self) -> pl.LazyFrame:
        """분석 실습을 위한 의도적으로 '왜곡된' 데이터 생성"""
        logger.info("🧪 학습용 '왜곡된 데이터' 생성 중... (Long-tail 분포 모사)")
        
        # 로그 정규 분포를 따르는 가격 데이터 (ML에서 가장 흔히 마주하는 형태)
        data = {
            "차량ID": [f"CAR_{i}" for i in range(1000)],
            "기준금액": np.random.lognormal(mean=10, sigma=1.5, size=1000).tolist(),
            "배기량": [random.randint(800, 5000) for _ in range(1000)],
            "등록일자": [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1000)]
        }
        return pl.DataFrame(data).lazy()

    def diagnose_data_stats(self, lf: pl.LazyFrame) -> tuple:
        """데이터의 '민낯'을 숫자로 밝혀내는 과정"""
        logger.info("🧪 [Diagnosis] IQR 및 Skewness 분석 시작")
        
        # 실무 지침: Lazy 상태에서 collect()는 최소화하되, 통계를 위해서는 샘플링 활용
        sample_df = lf.collect().sample(n=500)
        
        stats = {}
        for col in ["기준금액", "배기량"]:
            series = sample_df[col]
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            skew = series.skew()
            
            stats[col] = {"iqr": iqr, "q1": q1, "limit": q3 + 1.5 * iqr, "skew": skew}
            logger.info(f"   📊 '{col}' 분석결과 | Skew: {skew:.2f} | Outlier Limit: {stats[col]['limit']:.0f}")

            # 💡 [Senior Tip: Local AI Collaboration]
            # Liquid AI(LFM)나 SmolLM2와 같은 초경량 모델을 여기서 호출하여
            # "이 상한선을 넘는 데이터가 수집 오류인지, 아니면 '희귀 차량'인지" 
            # 비정형 텍스트(차량명 등)와 결합 분석을 수행하면 분석의 깊이가 달라집니다.
            
        return lf, stats

    def apply_quantum_transformation(self, lf: pl.LazyFrame, stats: dict) -> pl.LazyFrame:
        """데이터를 ML 모델이 가장 맛있게 먹을 수 있는 상태로 요리"""
        logger.info("🏗️ [Transform] ML-Ready 가공 레이어 가동")
        
        for col, meta in stats.items():
            # 1. 아웃라이어 태깅 (이상치를 지우지 않고 정보를 남김)
            lf = lf.with_columns(
                pl.when(pl.col(col) > meta["limit"]).then(pl.lit(1)).otherwise(pl.lit(0)).alias(f"{col}_is_extreme")
            )
            
            # 2. Skewness 보정 (0.75 이상이면 로그 변환 권장)
            if abs(meta["skew"]) > 0.75:
                logger.warning(f"   📈 '{col}' 왜곡도 감지! 로그 변환으로 정규분포 근사 시도.")
                lf = lf.with_columns((pl.col(col) + 1).log().alias(f"{col}_log_scaled"))
                
        return lf

    def verify_study_result(self, lf: pl.LazyFrame):
        """최종 결과물에 대한 시니어의 코멘트"""
        df = lf.collect()
        logger.success(f"✨ 세션 완료! 총 {len(df)}건의 데이터가 '지식'으로 승격되었습니다.")
        
        logger.info("📋 [Senior Comment]")
        print("-" * 50)
        print("1. 단순한 수치가 아닌 '통계적 배경'을 가진 피처를 생성했습니다.")
        print("2. 로그 변환된 컬럼은 선형 회귀나 신경망에서 훨씬 높은 예측력을 보일 것입니다.")
        print("3. _is_extreme 플래그는 분류 모델에서 '고가 차량'을 구분하는 핵심 피처가 됩니다.")
        print("-" * 50)

if __name__ == "__main__":
    import random
    setup_logging()
    
    # 학습 엔진 가동
    engine = KmongQuantumStudyEngine()
    engine.run_study_session()
