"""
더미 데이터 생성기
==================
실제 고객 데이터 대신 AI에게 질문할 때 사용할 샘플 데이터 생성

Author: Antigravity AI (Security Mode)
Date: 2026-01-22
"""

import polars as pl
from datetime import datetime, timedelta
import random


def create_dummy_vehicle_data(num_rows: int = 10) -> pl.DataFrame:
    """차량 등록 데이터 구조를 모방한 더미 데이터 생성"""
    
    # 더미 데이터 생성
    data = {
        "등록일": [
            datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))
            for _ in range(num_rows)
        ],
        "차량명": [f"차량{i}" for i in range(num_rows)],
        "제조사": random.choices(["제조사A", "제조사B", "제조사C"], k=num_rows),
        "배기량": [random.randint(1000, 3000) for _ in range(num_rows)],
        "회원구분명": random.choices(["개인", "법인", "개인사업자"], k=num_rows),
        "용도": random.choices(["자가용", "영업용", "관용"], k=num_rows),
        "지역": random.choices(["서울", "경기", "인천", "부산"], k=num_rows),
        "취득금액": [random.randint(1000, 5000) * 10000 for _ in range(num_rows)],
        "번호판": [
            random.choice(["12가1234", "34나5678", "56하7890", "78바9012"])
            for _ in range(num_rows)
        ],
    }
    
    df = pl.DataFrame(data)
    
    return df


def main():
    """더미 데이터 생성 및 저장"""
    print("🔒 더미 데이터 생성 중...")
    
    # 더미 데이터 생성
    df_dummy = create_dummy_vehicle_data(num_rows=10)
    
    # CSV로 저장 (AI에게 질문할 때 사용)
    output_path = "./data/sample_structure.csv"
    df_dummy.write_csv(output_path)
    
    print(f"✅ 더미 데이터 생성 완료: {output_path}")
    print(f"   - 행 수: {len(df_dummy)}")
    print(f"   - 컬럼 수: {len(df_dummy.columns)}")
    print("\n📋 컬럼 구조:")
    print(df_dummy.head(3))
    
    print("\n💡 사용 방법:")
    print("   1. AI에게 질문할 때 이 파일 사용")
    print("   2. 실제 데이터는 절대 AI에게 전달 금지")
    print("   3. 구조만 확인하면 충분!")


if __name__ == "__main__":
    main()
