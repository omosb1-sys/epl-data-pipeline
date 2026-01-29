
# ⚡ SKILL: Polars High-Performance Engineer (8GB RAM Optimization)

> **"Pandas is dead (for Big Data on Mac)."**  
> 8GB RAM 환경에서 수백만~수천만 건의 데이터를 처리하기 위한 **Polars 최적화(Lazy Evaluation) 전략**입니다.

## 1. Golden Rules (절대 수칙)
*   **Always Lazy (`.lazy()`)**: 데이터를 메모리에 즉시 로드(`read_csv`)하지 말고, 반드시 **스캔(`scan_csv`/`scan_parquet`)** 후 `LazyFrame` 상태에서 조작한다.
*   **One `collect()`**: 연산 파이프라인의 **가장 마지막**에 단 한 번만 `collect()`를 호출하여 메모리 스파이크를 방지한다.
*   **Streaming Mode**: `collect(streaming=True)` 옵션을 기본값으로 사용한다. (청크 단위 처리)
*   **Select First**: 모든 쿼리의 첫 줄은 불필요한 컬럼을 날리는 `select()`여야 한다.

## 2. Memory Logic (메모리 절약 패턴)
*   **String -> Categorical**: 고유값(Cardinality)이 적은 문자열 컬럼(팀명, 리그명 등)은 로드 즉시 `.cast(pl.Categorical)`로 변환한다. (메모리 80% 절감)
*   **Int64 -> Int32/16**: 불필요하게 큰 정수형은 `pl.Int32` 또는 `pl.Int16`으로 다운캐스팅한다.
*   **Sink to Disk**: 결과물이 너무 클 경우 `collect()` 대신 `sink_parquet("output.parquet")`를 사용하여 디스크에 바로 쓴다.

## 3. Code Snippet (Optimization Template)

### 3.1 The "Lazy" Pipeline
```python
import polars as pl

# 1. Scan (Not Read)
q = (
    pl.scan_parquet("huge_data.parquet")
    # 2. Filter Early (Predicate Pushdown)
    .filter(pl.col("date") > "2024-01-01")
    # 3. Optimize Types
    .with_columns([
        pl.col("team_name").cast(pl.Categorical),
        pl.col("goals").cast(pl.Int8)
    ])
    # 4. Complex Logic (Lazy)
    .group_by("team_name")
    .agg([
        pl.col("goals").sum().alias("total_goals"),
        pl.col("xg").mean().alias("avg_xg")
    ])
)

# 5. Execute with Streaming (Memory Safe)
df_result = q.collect(streaming=True)
```

### 3.2 Zero-Copy CSV to Parquet
*   CSV 분석은 느리다. 들어오자마자 Parquet으로 변환해라.
```python
# Streaming convert
pl.scan_csv("raw_log.csv").sink_parquet("optimized_log.parquet")
```
