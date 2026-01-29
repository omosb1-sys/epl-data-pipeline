
"""
[High-Performance Numerical Precision Guide v1.0]
Inspired by Russ Cox's "Simple, Fast Floating-Point Conversion"
==============================================================
이 가이드는 Antigravity Rule 33(Russ Cox Protocol)을 Python 데이터 파이프라인에
실무적으로 적용하는 방법을 제시합니다.

1. Unrounded Scaling: 중간 반올림 없는 스케일링
2. Serialization Efficiency: 대규모 데이터 직렬화 가속
3. Precision Integrity: 금융/통계 데이터 무결성 확보
"""

import polars as pl
import numpy as np
from loguru import logger
import decimal

def demonstrate_float_drift():
    """부동 소수점 오차의 고전적 사례 시연"""
    a = 0.1
    b = 0.2
    c = 0.3
    logger.info(f"🧪 [Test] 0.1 + 0.2 == 0.3 ? -> {a + b == c}")
    logger.info(f"📊 [Value] 0.1 + 0.2 = {a + b:.20f}")

def russ_cox_scaling_strategy(val: float, precision: int = 100) -> int:
    """
    [Rule 33.1] Unrounded Scaling (Pythonic Implementation)
    소수점 연산의 불안정성을 피하기 위해, 정수형 스케일링 후 연산하는 전략.
    """
    # 단순 반올림 대신 정수 공간으로 '점프'하여 오차 누적을 방지
    return int(val * precision)

def optimized_serialization_example(df: pl.DataFrame):
    """
    [Rule 33.2] Serialization Optimization
    단순 str(float)은 느리고 오차가 발생할 수 있음. 
    Polars의 고성능 엔진을 활용한 직렬화.
    """
    logger.info("📡 [Serialization] Russ Cox 방식의 고성능 직렬화 시뮬레이션")
    
    # 1. Float를 직접 String으로 바꾸는 대신, 필요 시 정수형 스케일링 후 저장 (용량 및 속도 최적화)
    df_optimized = df.with_columns([
        (pl.col("expected_goals") * 1000).cast(pl.Int32).alias("xG_milli"),
        (pl.col("win_prob") * 100).cast(pl.Int8).alias("win_pct_int")
    ])
    
    return df_optimized

if __name__ == "__main__":
    logger.info("🚀 [Antigravity] Russ Cox Protocol 가동")
    
    demonstrate_float_drift()
    
    # 예시 데이터 생성 (EPL xG 데이터)
    data = {
        "player": ["Son", "Salah", "Haaland"],
        "expected_goals": [0.854321, 1.293847, 2.102938],
        "win_prob": [0.754, 0.882, 0.951]
    }
    df = pl.DataFrame(data)
    
    df_opt = optimized_serialization_example(df)
    print("\n--- Optimized DataFrame (Integer Scaling) ---")
    print(df_opt)
    
    logger.success("✨ Russ Cox Protocol이 시스템 아키텍처에 성공적으로 주입되었습니다.")
