import pandas as pd
import glob
import os

# 1. 파일 경로 설정
data_dir = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/naver_shopping_analysis/data"
csv_files = glob.glob(os.path.join(data_dir, "naver_shopping_*.csv"))

print(f"✅ 발견된 파일 개수: {len(csv_files)}")

# 2. 데이터 통합 및 로드 (BOM 대응을 위해 utf-8-sig 사용)
df_list = []
for file in csv_files:
    temp_df = pd.read_csv(file, encoding='utf-8-sig')
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

# 3. 기본 정보 출력
print("\n--- 데이터 요약 ---")
print(f"전체 행 개수: {len(df)}")
print(f"컬럼 정보: {df.columns.tolist()}")
print(df.head())

# 4. 분석 인사이트 가능성 체크 (가격 결측치 등)
print("\n--- 분석 품질 체크 ---")
print(df[['lprice', 'brand', 'mall_name']].describe(include='all'))
