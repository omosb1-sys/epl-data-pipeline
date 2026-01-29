"""
[Kmong Project] Pipeline Verification & Dummy Data Generator
=============================================================
이 스크립트는 V5 파이프라인의 보안 및 성능을 검증하기 위한 '더미 엑셀' 생성 및 실행 테스트를 수행합니다.
Rule 15.1 (Dummy Data First) 준수.
"""

import polars as pl
import pandas as pd
from pathlib import Path
from loguru import logger
import random
from datetime import datetime, timedelta

def generate_complex_dummy_excel(output_dir: Path, num_files: int = 2, rows_per_file: int = 100):
    """실제 kmong 엑셀 구조를 모방한 더미 데이터 생성"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    brands = ["현대", "기아", "제네시스", "BMW", "벤츠", "아우디", "테슬라"]
    classes = ["개인", "법인", "관용", "개인사업자"]
    
    for i in range(num_files):
        data = {
            "등록일자": [(datetime.now() - timedelta(days=random.randint(0, 500))).strftime("%Y-%m-%d") for _ in range(rows_per_file)],
            "소유자명": [f"홍길동_{random.randint(1000, 9999)}" for _ in range(rows_per_file)],
            "전화번호": [f"010-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}" for _ in range(rows_per_file)],
            "차대번호": [f"ABC{random.randint(100000, 999999)}XYZ" for _ in range(rows_per_file)],
            "주소": [f"서울특별시 강남구 테헤란로 {random.randint(1, 500)}번지" for _ in range(rows_per_file)],
            "제조사명": [random.choice(brands) for _ in range(rows_per_file)],
            "차량명": [f"모델_{random.randint(1, 10)}" for _ in range(rows_per_file)],
            "회원구분명": [random.choice(classes) for _ in range(rows_per_file)],
            "기준금액": [random.randint(1000, 9000) * 10000 for _ in range(rows_per_file)],
            "취득금액": [random.randint(1000, 9000) * 10000 for _ in range(rows_per_file)]
        }
        
        df = pd.DataFrame(data)
        file_path = output_dir / f"Kmong_Sample_Data_{i}.xlsx"
        df.to_excel(file_path, index=False, engine='openpyxl')
        logger.info(f"✨ Dummy Excel Generated: {file_path}")

def run_pipeline_test():
    """V5 파이프라인 통합 테스트 실행"""
    from kmong_production_pipeline_v5 import KmongProductionPipeline, setup_logging
    
    setup_logging()
    logger.info("🧪 [Test] V5 Pipeline Integration Test Starting...")
    
    # 파이프라인 가동
    engine = KmongProductionPipeline()
    engine.run()
    
    # 결과 확인
    master_path = Path("data/kmong_project/processed_v5/master_data_v5.parquet")
    if master_path.exists():
        df = pl.read_parquet(master_path)
        logger.success(f"📈 Test Success! Master Shape: {df.shape}")
        
        # PII 마스킹 검증 결과 출력
        logger.info("🔒 Security Validation Sample:")
        print(df.select(["소유자명", "주소", "차대번호"]).head(5))
        
        # 비즈니스 로직 검증 결과 출력
        if "Market_Segment" in df.columns:
            logger.info("📊 Business Logic (Segmentation) Sample:")
            print(df.select(["제조사명", "기준금액", "Market_Segment"]).head(5))
    else:
        logger.error("🛑 Test Failed: Master file not found.")

if __name__ == "__main__":
    raw_dir = Path("data/kmong_project/raw")
    # 1. 더미 데이터 생성
    generate_complex_dummy_excel(raw_dir)
    
    # 2. 파이프라인 테스트
    run_pipeline_test()
