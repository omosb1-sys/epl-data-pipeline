import pandas as pd
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# 1. 모델 재학습 (환경 복원)
data_dir = "./naver_shopping_analysis/data"
csv_files = glob.glob(os.path.join(data_dir, "naver_shopping_*.csv"))
df = pd.concat([pd.read_csv(f, encoding='utf-8-sig') for f in csv_files], ignore_index=True)

df = df.drop_duplicates(subset=['product_id']).copy()
df['lprice'] = pd.to_numeric(df['lprice'], errors='coerce')
df = df.dropna(subset=['lprice', 'title'])

# 아웃라이어 제거
lower = df['lprice'].quantile(0.01)
upper = df['lprice'].quantile(0.99)
df_clean = df[(df['lprice'] >= lower) & (df['lprice'] <= upper)].copy()

# 텍스트 전처리
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^가-힣a-z0-9\s]', ' ', text)
    return text

df_clean['clean_title'] = df_clean['title'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3)
X = vectorizer.fit_transform(df_clean['clean_title'])
y = np.log1p(df_clean['lprice'])

model = Ridge(alpha=1.0)
model.fit(X, y)

# 2. A/B 테스트 시뮬레이션 로직
# 타겟: 스탠리, 스타벅스 등 브랜드 제품 100개 샘플링
target_brands = ['스탠리', '스타벅스', '써모스', '텀블러']
mask = df_clean['title'].apply(lambda x: any(b in x for b in target_brands))
sample_df = df_clean[mask].sample(100, random_state=42).copy()

def optimize_title(title):
    new_title = title
    
    # 전략 1: "Fake Premium" 제거 (스텐, 이중진공, 가성비)
    removals = ['스텐', '스테인리스', '이중진공', '가성비', '실속', '저렴한', '특가']
    for r in removals:
        new_title = new_title.replace(r, '')
        
    # 전략 2: 용량 표기 뒤로 이동 (단순 regex)
    # 예: "500ml 스탠리" -> "스탠리 ... 500ml"
    # (여기서는 간단히 용량 패턴을 제거하고 끝에 붙이는 식으로 시뮬레이션)
    cap_pattern = r'(\d+ml|\d+\.\d+L|\d+L)'
    match = re.search(cap_pattern, new_title)
    capacity = ""
    if match:
        capacity = match.group(0)
        new_title = re.sub(cap_pattern, '', new_title)
        
    # 전략 3: 가치 키워드 전진 배치 (한정판, 정품 등)
    # 브랜드가 있다면 그 앞에 "정품", 뒤에 "에디션" 등을 맥락에 맞게 추가 (시뮬레이션용 강제 주입)
    # 실제로는 제품이 정품/에디션이어야 하므로, 여기서는 '브랜딩 최적화'를 가정하고 키워드 보정
    
    prefix = ""
    if "스탠리" in new_title or "스타벅스" in new_title:
        if "정품" not in new_title:
            prefix += " [본사정품]"
    
    suffix = ""
    if capacity:
        suffix += f" {capacity}"
        
    final_title = f"{prefix} {new_title} {suffix}".strip()
    # 공백 정리
    final_title = re.sub(r'\s+', ' ', final_title)
    return final_title

# 변형 그룹(Variant) 생성
sample_df['optimized_title'] = sample_df['title'].apply(optimize_title)

# 가격 예측
vec_original = vectorizer.transform(sample_df['title'].apply(clean_text))
vec_optimized = vectorizer.transform(sample_df['optimized_title'].apply(clean_text))

pred_original = np.expm1(model.predict(vec_original))
pred_optimized = np.expm1(model.predict(vec_optimized))

sample_df['pred_original'] = pred_original
sample_df['pred_optimized'] = pred_optimized
sample_df['lift'] = (sample_df['pred_optimized'] - sample_df['pred_original']) / sample_df['pred_original'] * 100

# 3. 결과 시각화
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 3-1. Before & After Price Distribution
plt.figure(figsize=(10, 6))
sns.kdeplot(sample_df['pred_original'], fill=True, label='Original (A)', color='gray')
sns.kdeplot(sample_df['pred_optimized'], fill=True, label='Optimized (B)', color='red')
plt.title('A/B 테스트 시뮬레이션: 상품명 최적화에 따른 가치(가격) 변화')
plt.xlabel('AI 예측 적정 가격 (Predicted Price)')
plt.legend()
plt.savefig('ab_test_distribution.png')

# 3-2. Lift Chart (상승폭)
plt.figure(figsize=(10, 6))
sns.histplot(sample_df['lift'], bins=20, color='orange', kde=True)
plt.axvline(x=0, color='black', linestyle='--')
plt.title('가격 상승 잠재력(Lift %) 분포')
plt.xlabel('예상 가격 상승률 (%)')
plt.savefig('ab_test_lift.png')

# 4. 결과 요약 출력
print(f"📊 A/B 테스트 시뮬레이션 결과 (n=100)")
print(f"평균 원본 예측가: {sample_df['pred_original'].mean():,.0f}원")
print(f"평균 최적화 예측가: {sample_df['pred_optimized'].mean():,.0f}원")
print(f"💰 평균 상승률 (Lift): +{sample_df['lift'].mean():.2f}%")
print(f"✅ 가격 상승 케이스 비율: {(sample_df['lift'] > 0).mean()*100:.1f}%")

print("\n[변화 예시 TOP 3]")
for i, row in sample_df.sort_values(by='lift', ascending=False).head(3).iterrows():
    print(f"Before: {row['title']}")
    print(f"After : {row['optimized_title']}")
    print(f"Change: {row['pred_original']:,.0f}원 -> {row['pred_optimized']:,.0f}원 (+{row['lift']:.1f}%)\n")
