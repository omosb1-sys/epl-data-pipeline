import polars as pl
import numpy as np
from pathlib import Path

# [Senior Analysis] ChartGPU의 진정한 성능을 보여주기 위해 5만 건의 더미 데이터를 생성합니다.
# 실제 의뢰인 데이터를 받기 전, 시스템의 한계치를 테스트하기 위함입니다.

def generate_big_dummy_data(output_path: str, n_rows=50000):
    np.random.seed(42)
    
    data = {
        "데이터 연도 (YYYY)": np.random.choice(["2024", "2025"], n_rows),
        "데이터 월 (MM)": np.random.choice([str(i).zfill(2) for i in range(1, 13)], n_rows),
        "차종명": np.random.choice(["승용", "승합", "화물"], n_rows),
        "배기량": (np.random.normal(2000, 800, n_rows).clip(800, 5000)).astype(int).astype(str),
        "차명": np.random.choice(["Grandeur", "Avante", "Sonata", "K5", "G80"], n_rows),
        "지역": np.random.choice(["서울", "경기", "부산", "인천"], n_rows),
        "취득금액": (np.random.lognormal(17, 0.5, n_rows)).astype(int).astype(str),
        "고객유형_분류": np.random.choice(["개인", "법인"], n_rows, p=[0.7, 0.3]),
        "법인세부_분류": np.random.choice(["일반", "Rental", "Taxi"], n_rows, p=[0.8, 0.15, 0.05])
    }
    
    df = pl.DataFrame(data)
    
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_p)
    print(f"✅ {n_rows:,} 행의 테스트 마스터 데이터를 생성했습니다: {output_p}")

if __name__ == "__main__":
    generate_big_dummy_data("./data/kmong_project/processed/master_v1.parquet", 50000)
