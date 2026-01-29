"""
크몽 샘플 파일 분석기 (Red Team Hardened)
=========================================
- Path Independence: 실행 위치에 상관없이 동작
- Standardized Logging: 시니어 분석가용 로깅 적용

Author: Antigravity AI
Date: 2026-01-24
"""

import polars as pl
import openpyxl
from pathlib import Path
import os

def analyze_excel_structure(file_path: Path):
    """엑셀 파일 구조 분석"""
    print("=" * 60)
    print("📊 크몽 샘플 파일 분석 시작 [Red Team Version]")
    print("=" * 60)
    
    if not file_path.exists():
        print(f"❌ 에러: 파일을 찾을 수 없습니다: {file_path}")
        return None

    # 1. 파일 기본 정보
    file_size = file_path.stat().st_size / 1024  # KB
    print(f"\n📁 파일 정보:")
    print(f"   - 파일명: {file_path.name}")
    print(f"   - 크기: {file_size:.1f} KB")
    
    # 2. 시트 구조 확인
    print(f"\n📋 시트 구조 분석:")
    try:
        wb = openpyxl.load_workbook(str(file_path), read_only=True)
        sheet_names = wb.sheetnames
        print(f"   - 총 시트 수: {len(sheet_names)}")
        wb.close()
    except Exception as e:
        print(f"   ⚠️ 시트 읽기 오류: {e}")
    
    # 3. 데이터 샘플 확인 (Pandas -> Polars)
    try:
        import pandas as pd
        pdf = pd.read_excel(str(file_path), sheet_name=0)
        df = pl.from_pandas(pdf)
        print(f"\n📊 데이터 샘플 (첫 번째 시트): {len(df):,}행 x {len(df.columns)}열")
        print(df.head(3))
        return df
    except Exception as e:
        print(f"   ❌ 데이터 로드 오류: {e}")
        return None

def main():
    # [Red Team] 경로 유연성 확보
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # 기본 경로 설정
    file_path = project_root / "data" / "kmong_project" / "raw" / "sample.xlsm"
    
    analyze_excel_structure(file_path)

if __name__ == "__main__":
    main()
