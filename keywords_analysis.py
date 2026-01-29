import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 데이터 로드 및 전처리
data_dir = "./naver_shopping_analysis/data"
csv_files = glob.glob(os.path.join(data_dir, "naver_shopping_*.csv"))
df = pd.concat([pd.read_csv(f, encoding='utf-8-sig') for f in csv_files], ignore_index=True)

# 중복 제거 및 lprice 수치화
df = df.drop_duplicates(subset=['product_id']).copy()
df['lprice'] = pd.to_numeric(df['lprice'], errors='coerce')
df = df.dropna(subset=['lprice', 'title'])

# 2. 파생 컬럼 생성 (Feature Engineering)
# 주요 키워드 정의
keywords = {
    '대용량': ['대용량', '1L', '1000ml', '887ml', '퀜처', '대형'],
    '빨대/스트로': ['빨대', '스트로', 'straw'],
    '보온보냉': ['보온', '보냉', '진공'],
    '손잡이/핸들': ['손잡이', '핸들', 'handle'],
    '스텐': ['스텐', '스테인리스', 'stainless'],
    '브랜드_스탠리': ['스탠리', 'stanley'],
    '브랜드_써모스': ['써모스', 'thermos'],
    '브랜드_스타벅스': ['스타벅스', 'starbucks', '스벅']
}

for col, kws in keywords.items():
    df[col] = df['title'].str.lower().apply(lambda x: 1 if any(kw.lower() in x for kw in kws) else 0)

# 3. EDA 및 상관관계 분석
# 상관계수 계산 (수치형 컬럼들만)
feature_cols = list(keywords.keys()) + ['lprice']
corr_matrix = df[feature_cols].corr()

# 4. 시각화 (한글 폰트 설정 - Mac 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 시각화 1: 가격 분포와 키워드 유무 (대용량 여부)
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df[df['lprice'] < 100000], x='lprice', hue='대용량', fill=True)
plt.title('대용량 키워드 유무에 따른 가격 분포 (10만원 이하)')
plt.savefig('price_dist_capacity.png')

# 시각화 2: 상관관계 히트맵
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt='.2f', center=0)
plt.title('키워드 속성 및 가격 간 상관관계 히트맵')
plt.savefig('correlation_heatmap.png')

# 시각화 3: 키워드별 평균 가격 비교
mean_prices = {col: df[df[col] == 1]['lprice'].mean() for col in keywords.keys()}
price_series = pd.Series(mean_prices).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=price_series.index, y=price_series.values, palette='viridis')
plt.xticks(rotation=45)
plt.title('주요 속성(키워드)별 평균 최저가 비교')
plt.ylabel('평균 가격 (원)')
plt.savefig('avg_price_by_keyword.png')

print("✅ 분석 코드 실행 완료. 이미지 3개가 생성되었습니다.")
print("\n--- 주요 키워드별 상관계수 (가격 기준) ---")
print(corr_matrix['lprice'].sort_values(ascending=False))
